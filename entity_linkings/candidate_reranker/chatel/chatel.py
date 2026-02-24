from dataclasses import dataclass
from logging import getLogger
from typing import Optional

import torch
from datasets import Dataset

from entity_linkings.data_utils import cut_context_window
from entity_linkings.utils import BaseSystemOutput, calculate_top1_accuracy

from ..base import RerankerBase, RetrieverBase
from .gpt import OpenAI_API
from .utils import process_multi_choice_prompt

logger = getLogger(__name__)
logger.setLevel("INFO")


class CHATEL(RerankerBase):
    """ Entity Disambiguation via Fusion Entity Decoding
    """
    @dataclass
    class Config(RerankerBase.Config):
        model_name_or_path: str = "gpt-4o-mini"
        num_description_characters: int = 200
        max_generation_length: int = 100
        context_window_chars: int = 500
        temperature: float = 0.2
        top_p: float = 0.0
        seed: int = 0
        token: str = "<API_KEY>"
        organization_key: Optional[str] = None

    def __init__(self, retriever: RetrieverBase, config: Optional[Config] = None) -> None:
        super().__init__(retriever, config)
        self.model = OpenAI_API(
            model_name=self.config.model_name_or_path,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            seed=self.config.seed,
            max_generation_length=self.config.max_generation_length,
            token=self.config.token,
            organization_key=self.config.organization_key,
        )

    def first_step_prompt(self, sentence: str, start: int, end: int) -> str:
        mention = sentence[start:end]
        prompt_sentence, _, _ = cut_context_window(sentence, start, end, self.config.context_window_chars)
        return prompt_sentence + " \n What does " + mention + " in this sentence referring to?"

    def second_step_prompt(self, first_result: str, mention: str, candidates_ids: list[str]) -> str:
        multi_choice_prompt = ""
        candidates = [self.dictionary(c) for c in candidates_ids]
        for i, cand in enumerate(candidates):
            description = cand['name']
            if cand['description']:
                description += ' ' + cand['description'][:self.config.num_description_characters]
            multi_choice_prompt += f"({i+1}). {description}\n"
        multi_choice_prompt = first_result + '\n\n' + f'Which of the following does {mention} referes to?' + '\n\n' + multi_choice_prompt
        return multi_choice_prompt

    @torch.no_grad()
    def evaluate(self, dataset: Dataset, num_candidates: int = 30, batch_size: int = 32, **args: int) -> dict[str, float]:
        candidates = self.retriever.retrieve_candidates(
            dataset, only_negative=False, top_k=num_candidates, batch_size=batch_size,
        )

        first_prompts = []
        for text, entities in zip(dataset["text"], dataset["entities"]):
            for ent in entities:
                if not ent["label"]:
                    continue
                first_prompts.append(self.first_step_prompt(text, ent['start'], ent['end']))

        if isinstance(self.model, OpenAI_API):
            test_generation = self.model.generate(first_prompts[0])[0]
            estimated_cost = self.model.estimate(first_prompts, test_generation)
            logger.warning(f"Estimated cost for first step prompts: ${estimated_cost:.4f}")

        first_results = self.model.generate(first_prompts)

        second_prompts = []
        count = 0
        for text, entities in zip(dataset["text"], dataset["entities"]):
            for ent in entities:
                if not ent["label"]:
                    continue
                second_prompts.append(self.second_step_prompt(first_results[count], text[ent["start"]:ent["end"]], candidates[count]))
                count += 1

        if isinstance(self.model, OpenAI_API):
            test_generation = self.model.generate(second_prompts[0])[0]
            estimated_cost = self.model.estimate(second_prompts, test_generation)
            logger.warning(f"Estimated cost for second step prompts: ${estimated_cost:.4f}")

        second_results = self.model.generate(second_prompts)

        count = 0
        num_corrects, num_golds = 0, 0
        for entity in dataset["entities"]:
            for ent in entity:
                if not ent["label"]:
                    continue
                num_golds += 1
                result = second_results[count]
                choiced_id= process_multi_choice_prompt(result, candidates[count])
                pred = self.dictionary(candidates[count][choiced_id])['id'] if choiced_id != -1 else None
                if pred in ent["label"]:
                    num_corrects += 1
                count += 1
        return calculate_top1_accuracy(num_corrects, num_golds)

    @torch.no_grad()
    def predict(self, sentence: str, spans: list[tuple[int, int]], num_candidates: int = 30) -> list[BaseSystemOutput]:
        predictions = self.retriever.predict(sentence, spans, top_k=num_candidates)
        candidates = [[cand_id.id for cand_id in c] for c in predictions]

        first_prompts = []
        for b, e in spans:
            first_prompts.append(self.first_step_prompt(sentence, b, e))
        first_results = self.model.generate(first_prompts)

        second_prompts = []
        for i, (b, e) in enumerate(spans):
            second_prompts.append(self.second_step_prompt(first_results[i], sentence[b:e], candidates[i]))
        second_results = self.model.generate(second_prompts)

        all_result = []
        for i, result in enumerate(second_results):
            pred = process_multi_choice_prompt(result, candidates[i])
            if -1 == pred:
                logger.warning(f"CHATEL could not find a valid entity for the mention: {sentence[spans[i][0]:spans[i][1]]}")
                continue
            top1_entity = self.dictionary(candidates[i][pred])
            all_result.append(BaseSystemOutput(query=sentence[b:e], start=b, end=e, id=top1_entity['id']))
        return all_result

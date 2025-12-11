from dataclasses import dataclass
from logging import getLogger
from typing import Optional

import torch
from datasets import Dataset

from entity_linkings.utils import calculate_top1_accuracy

from ..base import EntityRetrieverBase, PipelineBase
from .gpt import OpenAI_API
from .utils import process_multi_choice_prompt

logger = getLogger(__name__)
logger.setLevel("INFO")


class CHATEL(PipelineBase):
    """ Entity Disambiguation via Fusion Entity Decoding
    """
    @dataclass
    class Config(PipelineBase.Config):
        model_name_or_path: str = "gpt-4o-mini"
        max_context_length: int = 4096
        max_generation_length: int = 30
        temperature: float = 0.2
        top_p: float = 0.0
        seed: int = 0
        token: str = "<API_KEY>"
        organization_key: Optional[str] = None

    def __init__(self, retriever: EntityRetrieverBase, config: Optional[Config] = None) -> None:
        self.retriever = retriever
        self.dictionary = retriever.dictionary
        self.config = config if config is not None else self.Config()

        self.model = OpenAI_API(
            model_name=self.config.model_name_or_path,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            seed=self.config.seed,
            max_token_length=self.config.max_context_length,
            token=self.config.token,
            organization_key=self.config.organization_key,
        )

    def first_step_prompt(self, sentence: str, mention: str) -> str:
        return sentence + " \n What does " + mention + " in this sentence referring to?"

    def second_step_prompt(self, first_result: str, mention: str, candidates_ids: list[str]) -> str:
        multi_choice_prompt = ""
        candidates = [self.dictionary(c) for c in candidates_ids]
        for i, cand in enumerate(candidates):
            description = cand['name'] + ': ' + cand['description']
            multi_choice_prompt += f"({i+1}). {description}\n"
        multi_choice_prompt = first_result + '\n\n' + f'Which of the following entities is {mention} in this sentence?' + '\n\n' + multi_choice_prompt
        return multi_choice_prompt

    @torch.no_grad()
    def evaluate(self, dataset: Dataset, num_candidates: int = 30, batch_size: int = 32, **args: int) -> dict[str, float]:
        dataset = dataset.select(range(100))
        candidates_ids = self.retriever.retrieve_candidates(dataset, top_k=num_candidates, only_negative=False, batch_size=batch_size)
        first_prompts = []
        for text, entities in zip(dataset["text"], dataset["entities"]):
            for ent in entities:
                if not ent["label"]:
                    continue
                first_prompts.append(self.first_step_prompt(text, text[ent["start"]:ent["end"]]))

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
                second_prompts.append(self.second_step_prompt(first_results[count], text[ent["start"]:ent["end"]], candidates_ids[count]))
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
                choiced_id= process_multi_choice_prompt(result, candidates_ids[count])
                pred = self.dictionary(candidates_ids[count][choiced_id])['id'] if choiced_id != -1 else None
                if pred in ent["label"]:
                    num_corrects += 1
                count += 1
        return calculate_top1_accuracy(num_corrects, num_golds)

    @torch.no_grad()
    def predict(self, sentence: str, spans: Optional[list[tuple[int, int]]] = None, num_candidates: int = 30) -> list[list[dict[str, float]]]:
        if not spans:
            raise ValueError("Spans must be provided for CHATEL prediction.")

        candidates = self.retriever.predict(sentence, spans, top_k=num_candidates)
        first_prompts = []
        for b, e in spans:
            first_prompts.append(self.first_step_prompt(sentence, sentence[b:e]))
        first_results = self.model.generate(first_prompts)

        second_prompts = []
        for i, (b, e) in enumerate(spans):
            second_prompts.append(self.second_step_prompt(first_results[i], sentence[b:e], [c['id'] for c in candidates[i]]))
        second_results = self.model.generate(second_prompts)

        all_result = []
        for i, result in enumerate(second_results):
            pred = process_multi_choice_prompt(result, [c['id'] for c in candidates[i]])
            if -1 == pred:
                logger.warning(f"CHATEL could not find a valid entity for the mention: {sentence[spans[i][0]:spans[i][1]]}")
                all_result.append([])
                continue
            all_result.append([candidates[i][pred]])
        return all_result

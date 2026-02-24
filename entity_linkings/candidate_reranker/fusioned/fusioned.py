import os
from dataclasses import dataclass
from logging import getLogger
from typing import Optional

import torch
from datasets import Dataset
from torch.utils.data import DataLoader, SequentialSampler
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed

from entity_linkings.trainer import EntityLinkingTrainer, TrainingArguments
from entity_linkings.utils import BaseSystemOutput, calculate_top1_accuracy

from ..base import RerankerBase, RetrieverBase
from .collator import CollatorForFusioned
from .preprocessor import FusionedPreprocessor
from .reader import FusionDecoder

logger = getLogger(__name__)
logger.setLevel("INFO")


class FUSIONED(RerankerBase):
    """ Entity Disambiguation via Fusion Entity Decoding
    """

    @dataclass
    class Config(RerankerBase.Config):
        model_name_or_path: str = "google/flan-t5-base"
        max_context_length: int = 128
        max_candidate_length: int = 50
        context_window_chars: int = 500
        use_checkpoint: bool = False
        num_beams: int = 3
        max_new_tokens: int = 200
        min_length: int = 1
        document_token: str = "<extra_id_0>"
        passage_token: str = "<extra_id_1>"
        title_token: str = "<extra_id_2>"
        description_token: str = "<extra_id_3>"
        entity_token: str = "<extra_id_4>"
        mention_token: str = "<extra_id_5>"
        ent_start_token: str = "<extra_id_6>"
        ent_end_token: str = "<extra_id_7>"
        nil_token: str = "[NIL]"

    def __init__(self, retriever: RetrieverBase, config: Optional[Config] = None) -> None:
        super().__init__(retriever, config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name_or_path)
        if os.path.exists(self.config.model_name_or_path):
            self.model = FusionDecoder.from_pretrained(self.config.model_name_or_path)
        else:
            self.tokenizer.add_special_tokens({"additional_special_tokens": [self.config.nil_token]})
            t5 = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_name_or_path)
            t5.resize_token_embeddings(len(self.tokenizer))
            self.model = FusionDecoder(t5.config)
            self.model.load_t5(t5.state_dict())
        self.model.set_checkpoint(self.config.use_checkpoint)

        self.preprocessor = FusionedPreprocessor(
            tokenizer = self.tokenizer,
            title_token=self.config.title_token,
            ent_start_token=self.config.ent_start_token,
            ent_end_token=self.config.ent_end_token,
            entity_token=self.config.entity_token,
            max_context_length=self.config.max_context_length,
            max_candidate_length=self.config.max_candidate_length,
            context_window_chars=self.config.context_window_chars,
        )
        self.dictionary = self.preprocessor.dictionary_preprocess(self.retriever.dictionary)

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        num_candidates: int = 30,
        training_args: Optional[TrainingArguments] = None
    ) -> dict[str, float]:
        if training_args is None:
            training_args = TrainingArguments()
        set_seed(training_args.seed)

        train_candidates = self.retriever.retrieve_candidates(
            train_dataset,
            only_negative=True,
            top_k=num_candidates,
            batch_size=training_args.per_device_eval_batch_size,
        )
        train_dataset = self.preprocessor.dataset_preprocess(train_dataset, train_candidates)

        if eval_dataset is not None:
            eval_candidates = self.retriever.retrieve_candidates(
                eval_dataset,
                only_negative=True,
                top_k=num_candidates,
                batch_size=training_args.per_device_eval_batch_size,
            )
            eval_dataset = self.preprocessor.dataset_preprocess(eval_dataset, eval_candidates)

        trainer = EntityLinkingTrainer(
            model=self.model,
            args=training_args,
            data_collator=CollatorForFusioned(
                self.tokenizer,
                dictionary=self.dictionary,
                train=True
            ),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint

        results = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.log_metrics("train", results.metrics)
        if training_args.output_dir is not None:
            self.model.save_pretrained(training_args.output_dir)
            self.tokenizer.save_pretrained(training_args.output_dir)
            trainer.save_state()
            trainer.save_metrics("train", results.metrics)
        return results

    @torch.no_grad()
    def evaluate(self, dataset: Dataset, num_candidates: int = 30, batch_size: int = 32, **args: int) -> dict[str, float]:
        candidates = self.retriever.retrieve_candidates(
            dataset, only_negative=False, top_k=num_candidates, batch_size=batch_size,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.model.to(device)
        processed_dataset = self.preprocessor.dataset_preprocess(dataset, candidates)

        dataloader = DataLoader(
            processed_dataset,
            batch_size=batch_size,
            sampler=SequentialSampler(processed_dataset),
            collate_fn=CollatorForFusioned(
                self.tokenizer,
                dictionary=self.dictionary,
                train=False,
            ),
        )

        num_corrects, num_golds = 0, 0
        pbar  = tqdm(total=len(dataloader), desc='Evaluate')
        for batch in dataloader:
            pbar.update()
            labels = batch.pop("labels") # (batch_size, n_candidates)
            batch = batch.to(device)
            generated_ids = self.model.generate(
                **batch,
                num_beams=self.config.num_beams,
                max_new_tokens=self.config.max_new_tokens,
                min_length=self.config.min_length,
            )
            generated_ids = generated_ids.to('cpu')
            for generated_id, label in zip(generated_ids, labels):
                gold = self.tokenizer.decode(label, skip_special_tokens=True)
                pred = self.tokenizer.decode(generated_id, skip_special_tokens=True)
                num_golds += 1
                num_corrects += 1 if pred == gold else 0
        pbar.close()
        metric = calculate_top1_accuracy(num_corrects, num_golds)
        return metric

    @torch.no_grad()
    def predict(self, sentence: str, spans: list[tuple[int, int]], num_candidates: int = 30) -> list[BaseSystemOutput]:
        predictions = self.retriever.predict(sentence, spans=spans, top_k=num_candidates)
        candidates = [[cand_id.id for cand_id in c] for c in predictions]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.model.to(device)

        all_result = []
        collator = CollatorForFusioned(self.tokenizer, dictionary=self.dictionary)
        for i, (b, e) in enumerate(spans):
            encoding = self.preprocessor.process_context(sentence, b, e, candidates[i])
            batch = collator([encoding])
            batch = batch.to(device)
            generated_ids = self.model.generate(
                **batch,
                num_beams=self.config.num_beams,
                max_new_tokens=self.config.max_new_tokens,
                min_length=self.config.min_length,
            )
            generated_ids = generated_ids.to('cpu')
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            candidate_titles = [self.dictionary(cand)['name'] for cand in candidates[i]]
            if generated_text not in candidate_titles:
                logger.warning(f"FUSIONED could not find a valid entity for the mention: {sentence[spans[i][0]:spans[i][1]]}")
                continue
            pred = candidate_titles.index(generated_text)
            top1_entity = self.dictionary(candidates[i][pred])
            all_result.append(BaseSystemOutput(query=sentence[b:e], start=b, end=e, id=top1_entity['id']))
        return all_result

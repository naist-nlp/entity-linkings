import os
from dataclasses import dataclass
from typing import Optional

import torch
from datasets import Dataset
from torch.utils.data import DataLoader, SequentialSampler
from tqdm.auto import tqdm
from transformers import AutoTokenizer, set_seed

from entity_linkings.data_utils import Preprocessor
from entity_linkings.trainer import EntityLinkingTrainer, TrainingArguments
from entity_linkings.utils import BaseSystemOutput, calculate_top1_accuracy

from ..base import RerankerBase, RetrieverBase
from ..utils import compute_metrics
from .collator import CollatorForCrossEncoder
from .encoder import Encoder


class CROSSENCODER(RerankerBase):
    """ BLINK: Scalable Zero-shot Entity Linking with Dense Entity Retrieval (https://aclanthology.org/2020.emnlp-main.519/)
    """

    @dataclass
    class Config(RerankerBase.Config):
        """ BLINK configuration
        """
        model_name_or_path: str = "google-bert/bert-base-uncased"
        ent_start_token: str = "[START_ENT]"
        ent_end_token: str = "[END_ENT]"
        entity_token: str = "[ENT]"
        nil_token: str = "[NIL]"
        pooling: str = 'first'
        max_context_length: int = 128
        max_candidate_length: int = 50
        context_window_chars: int = 500

    def __init__(self, retriever: RetrieverBase, config: Optional[Config] = None) -> None:
        super().__init__(retriever, config)
        if os.path.exists(self.config.model_name_or_path):
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name_or_path)
            self.model = Encoder.from_pretrained(self.config.model_name_or_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name_or_path)
            special_tokens = [self.config.ent_start_token, self.config.ent_end_token, self.config.entity_token, self.config.nil_token]
            self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
            self.model = Encoder(self.config.model_name_or_path)
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.preprocessor = Preprocessor(
            self.tokenizer, self.config.ent_start_token, self.config.ent_end_token,
            self.config.entity_token, self.config.max_context_length,
            self.config.max_candidate_length, self.config.context_window_chars
        )
        self.dictionary = self.preprocessor.dictionary_preprocess(self.dictionary)

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
            data_collator=CollatorForCrossEncoder(self.tokenizer, dictionary=self.dictionary, train=True),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
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

        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        processed_dataset = self.preprocessor.dataset_preprocess(dataset, candidates)
        dataloader = DataLoader(
            processed_dataset,
            batch_size=batch_size,
            sampler=SequentialSampler(processed_dataset),
            collate_fn=CollatorForCrossEncoder(
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
            _, scores = self.model(**batch) # (batch_size, n_candidates)
            scores = scores.to('cpu')
            preds = scores.argmax(axis=1)  # (batch_size, )
            num_corrects += (labels == preds).sum().item()
            num_golds += labels.size(0)
        pbar.close()
        metric = calculate_top1_accuracy(num_corrects, num_golds)
        return metric

    @torch.no_grad()
    def predict(self, sentence: str, spans: list[tuple[int, int]], num_candidates: int = 30) -> list[BaseSystemOutput]:
        predictions = self.retriever.predict(sentence, spans=spans, top_k=num_candidates)
        candidates = [[cand_id.id for cand_id in c] for c in predictions]

        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        collator = CollatorForCrossEncoder(self.tokenizer, dictionary=self.dictionary)
        all_result = []
        for i, (b, e) in enumerate(spans):
            encodings = self.preprocessor.process_context(sentence, b, e, candidates[i])
            batch = collator([encodings])
            batch = batch.to(device)
            _, scores = self.model(**batch) # (None, (n_candidates, ))
            scores = scores.to('cpu')
            pred = scores.argmax(dim=1)[0].item()
            top1_entity = self.dictionary(candidates[i][pred])
            all_result.append(BaseSystemOutput(query=sentence[b:e], start=b, end=e, id=top1_entity['id']))
        return all_result

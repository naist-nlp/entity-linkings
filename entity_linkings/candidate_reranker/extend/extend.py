from dataclasses import dataclass
from typing import Optional

import torch
from datasets import Dataset
from torch.utils.data import DataLoader, SequentialSampler
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer,
    LongformerForQuestionAnswering,
    set_seed,
)

from entity_linkings.trainer import EntityLinkingTrainer, TrainingArguments
from entity_linkings.utils import BaseSystemOutput, calculate_top1_accuracy

from ..base import RerankerBase, RetrieverBase
from .collator import CollatorForExtend
from .preprocessor import ExtendPreprocessor
from .utils import compute_metrics, select_indices


class EXTEND(RerankerBase):

    @dataclass
    class Config(RerankerBase.Config):
        '''EXTEND configuration
        Args:
            - model_name_or_path (str): Pre-trained model name or path for mention and entity encoders
            - add_nil_token (bool): Whether to add a special token for NIL entity
            - nil_token (str): Special token for NIL entity
            - max_context_length (int): Maximum sequence length for mention context
            - pooling (str): Pooling method for obtaining fixed-size representations ('cls' or 'mean')
            - batch_size (int): Batch size for encoding mentions and entities
        '''
        model_name_or_path: str = "allenai/longformer-large-4096"
        ent_start_token: str = "[START_ENT]"
        ent_end_token: str = "[END_ENT]"
        nil_token: str = "[NIL]"
        max_context_length: int = 128
        context_window_chars: int = 500
        candidate_separator: str = '*'
        attention_window: int = 64
        modify_global_attention: int = 2
        mode: str = "max-prod" # "max-prod" or "max-end", "max-start", "max"

    def __init__(self, retriever: RetrieverBase, config: Optional[Config] = None) -> None:
        super().__init__(retriever, config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name_or_path)
        model_config = AutoConfig.from_pretrained(
            self.config.model_name_or_path,
            attention_window=self.config.attention_window
        )
        self.model = LongformerForQuestionAnswering.from_pretrained(self.config.model_name_or_path, config=model_config)
        special_tokens = [self.config.ent_start_token, self.config.ent_end_token, self.config.nil_token]
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        if self.model.config.vocab_size != len(self.tokenizer):
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.preprocessor = ExtendPreprocessor(
            tokenizer = self.tokenizer,
            dictionary = self.retriever.dictionary,
            ent_start_token=self.config.ent_start_token,
            ent_end_token=self.config.ent_end_token,
            candidate_separator=self.config.candidate_separator,
            max_context_length=self.config.max_context_length,
        )

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

        train_dataset = self.preprocessor.dataset_preprocess(train_dataset, train_candidates, train=True)
        if eval_dataset is not None:
            eval_candidates = self.retriever.retrieve_candidates(
                eval_dataset,
                only_negative=True,
                top_k=num_candidates,
                batch_size=training_args.per_device_eval_batch_size,
            )
            eval_dataset = self.preprocessor.dataset_preprocess(eval_dataset, eval_candidates, train=True)

        trainer = EntityLinkingTrainer(
            model=self.model,
            args=training_args,
            data_collator=CollatorForExtend(
                self.tokenizer,
                modify_global_attention=self.config.modify_global_attention,
            ),
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

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.model.to(device)

        processed_dataset = self.preprocessor.dataset_preprocess(dataset, candidates, train=False)
        dataloader = DataLoader(
            processed_dataset,
            batch_size=batch_size,
            sampler=SequentialSampler(processed_dataset),
            collate_fn=CollatorForExtend(
                self.tokenizer,
                modify_global_attention=self.config.modify_global_attention,
            ),
        )

        pbar  = tqdm(total=len(dataloader), desc='Evaluate')
        start_probs, end_probs = [], []
        for batch in dataloader:
            pbar.update()
            batch = batch.to(device)
            outputs = self.model(**batch)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
            start_probs.extend(torch.softmax(start_logits, dim=-1).tolist()) # (batch_size, seq_length)
            end_probs.extend(torch.softmax(end_logits, dim=-1).tolist()) # ( batch_size, seq_length)
        pbar.close()

        assert len(start_probs) == len(processed_dataset)
        assert len(end_probs) == len(processed_dataset)
        num_golds = len(processed_dataset)
        num_corrects = 0
        for data, start_prob, end_prob in zip(processed_dataset, start_probs, end_probs):
            prob = torch.tensor([start_prob, end_prob])  # (2, seq_length)
            start_index, end_index = select_indices(
                mode=self.config.mode,
                possible_indices=data["candidates_offsets"],
                classification_probabilities=prob,
            )
            pred_index = data['candidates_offsets'].index([start_index, end_index])
            pred = data["candidates"][pred_index]
            if pred in data["labels"]:
                num_corrects += 1
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
        collator = CollatorForExtend(self.tokenizer, modify_global_attention=self.config.modify_global_attention)
        for i, (b, e) in enumerate(spans):
            processed_text = self.preprocessor.process_context(sentence, b, e, candidates[i])
            encodings = collator([processed_text])
            encodings = encodings.to(device)
            outputs = self.model(**encodings)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
            scores = torch.stack([start_logits, end_logits], dim=0) # (2, batch_size, seq_length)
            probabilities = torch.softmax(scores, dim=-1) # (2, batch_size, seq_length)
            start_index, end_index = select_indices(
                mode=self.config.mode,
                possible_indices=processed_text['candidates_offsets'],
                classification_probabilities=probabilities[:, 0],
            )
            pred = processed_text['candidates_offsets'].index((start_index, end_index))
            top1_entity = self.dictionary(candidates[i][pred])
            all_result.append(BaseSystemOutput(query=sentence[b:e], start=b, end=e, id=top1_entity['id']))
        return all_result

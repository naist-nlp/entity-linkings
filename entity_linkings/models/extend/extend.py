import random
from dataclasses import dataclass
from typing import Any, Iterator, Optional

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader, SequentialSampler
from tqdm.auto import tqdm
from transformers import BatchEncoding, EvalPrediction, set_seed

from entity_linkings.data_utils import CollatorForExtend, truncate_around_mention
from entity_linkings.dataset.utils import preprocess
from entity_linkings.trainer import EntityLinkingTrainer, TrainingArguments
from entity_linkings.utils import calculate_top1_accuracy

from ..base import EntityRetrieverBase, PipelineBase
from .qa_span_classifier import QASpanClassifier
from .utils import (
    compute_char_to_tokens,
    process_candidates,
    select_indices,
)


class EXTEND(PipelineBase):

    @dataclass
    class Config(PipelineBase.Config):
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
        candidate_separator: str = '*'
        attention_window: Optional[int] = 64
        modify_global_attention: Optional[int] = 2
        mode: str = "max-prod" # "max-prod" or "max-end", "max-start", "max"

    def __init__(self, retriever: EntityRetrieverBase, config: Optional[Config] = None) -> None:
        super().__init__(retriever, config)
        special_tokens = [self.config.ent_start_token, self.config.ent_end_token, self.config.nil_token]
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        self.model = QASpanClassifier(
            model_name_or_path=self.config.model_name_or_path,
            attention_window=self.config.attention_window,
            modify_global_attention=self.config.modify_global_attention
        )
        self.model.resize_token_embeddings(len(self.tokenizer))

    def compute_metrics(self, p: EvalPrediction) -> dict[str, float]:
        print(p)
        loss, (_, scores) = p.predictions

        labels = p.label_ids # (2, batch_size, n_samples // num_entities)
        gold_start_positions = labels[0].ravel() # (n_samples, )
        gold_end_positions = labels[1].ravel() # (n_samples, )

        preds = scores.argmax(axis=-1)  # (2, batch_size, n_samples // num_entities)
        start_predictions = preds[0].ravel() # (n_samples, )
        end_predictions = preds[1].ravel() # (n_samples, )
        correct_full_predictions = np.logical_and(
            np.equal(start_predictions, gold_start_positions),
            np.equal(end_predictions, gold_end_positions),
        )
        accuracy = correct_full_predictions.mean().item()
        return {"loss": loss.sum(), "accuracy": accuracy}

    def data_filter(self, dataset: Dataset) -> Dataset:
        def _preprocess_example(examples: Dataset) -> Iterator[dict]:
            for text, entities in zip(examples["text"], examples["entities"]):
                for ent in entities:
                    if not ent["label"]:
                        continue
                    yield {"text": text, "start": ent["start"], "end": ent["end"], "label": ent["label"]}
        return preprocess(dataset, _preprocess_example)

    def process_context(self, text: str, start: int, end: int, candidate: list[str], labels: Optional[list[str]] = None) -> BatchEncoding:
        available_length = self.config.max_context_length - 2 # for [CLS] and [SEP] tokens
        head = self.tokenizer.encode(text[: start] + self.config.ent_start_token, add_special_tokens=False)
        mention = self.tokenizer.encode(text[start: end], add_special_tokens=False)
        tail = self.tokenizer.encode(self.config.ent_end_token + text[end:], add_special_tokens=False)

        input_ids = head + mention + tail
        input_ids, _, _ = truncate_around_mention(input_ids, available_length, len(head), len(head) + len(mention))
        assert len(input_ids) <= available_length

        if labels:
            gold_titles = [self.dictionary(labels[0])['name']]
            candidate = candidate[:-1] + [labels[0]]
            random.shuffle(candidate)
        else:
            gold_titles = []

        candidate_titles = [self.dictionary(cand)['name'] for cand in candidate]
        candidate_context, answer_starts, answer_ends, candidates_offsets = process_candidates(
            candidate_titles, gold_titles, separator=self.config.candidate_separator
        )

        encodings = self.tokenizer(
            self.tokenizer.decode(input_ids), candidate_context,
            return_offsets_mapping=True,
        )
        char2token = compute_char_to_tokens(
            candidate_context,
            [p == 1 for p in encodings.sequence_ids()],
            encodings["offset_mapping"]
        )
        encodings["candidates_offsets"] = [
            (char2token[si], char2token[ei - 1] + 1)
            for si, ei in candidates_offsets
        ]
        if labels:
            encodings["labels"] = [
                (char2token[ans_start], char2token[ans_end - 1] + 1)
                for ans_start, ans_end in zip(answer_starts, answer_ends)
            ]
        return encodings

    def data_preprocess(self, dataset: Dataset, train: bool = False) -> Dataset:
        def _preprocess_example(examples: Dataset) -> Iterator[BatchEncoding]:
            for text, start, end, candidates, labels in zip(examples['text'], examples['start'], examples['end'], examples['candidates'], examples['label']):
                if train:
                    encodings = self.process_context(text, start, end, candidates, labels)
                else:
                    encodings = self.process_context(text, start, end, candidates)
                yield encodings
        return preprocess(dataset, _preprocess_example)

    def train(
            self,
            train_dataset: Dataset,
            eval_dataset: Optional[Dataset] = None,
            num_candidates: int = 30,
            training_args: Optional[TrainingArguments] = None,
        ) -> dict[str, float]:
        if training_args is None:
            training_args = TrainingArguments()
        set_seed(training_args.seed)

        train_candidates = self.retriever.retrieve_candidates(
            train_dataset,
            only_negative=True,
            top_k=num_candidates,
            batch_size=training_args.per_device_eval_batch_size
        )
        filtered_train_dataset = self.data_filter(train_dataset)
        assert len(filtered_train_dataset) == len(train_candidates)
        filtered_train_dataset = filtered_train_dataset.add_column("candidates", train_candidates)
        processed_train_dataset = self.data_preprocess(filtered_train_dataset, train=True)
        if eval_dataset is not None:
            eval_candidates = self.retriever.retrieve_candidates(
                eval_dataset,
                only_negative=True,
                top_k=num_candidates,
                batch_size=training_args.per_device_eval_batch_size
            )
            filtered_eval_dataset = self.data_filter(eval_dataset)
            assert len(filtered_eval_dataset) == len(eval_candidates)
            filtered_eval_dataset = filtered_eval_dataset.add_column("candidates", eval_candidates)
            processed_eval_dataset = self.data_preprocess(filtered_eval_dataset, train=True)

        trainer = EntityLinkingTrainer(
            model=self.model,
            args=training_args,
            data_collator=CollatorForExtend(self.tokenizer),
            train_dataset=processed_train_dataset,
            eval_dataset=processed_eval_dataset if eval_dataset is not None else None,
            compute_metrics=self.compute_metrics,
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
        self.model.eval()
        candidates = self.retriever.retrieve_candidates(
            dataset,
            only_negative=False,
            top_k=num_candidates,
            batch_size=batch_size
        )
        filtered_dataset = self.data_filter(dataset)
        assert len(filtered_dataset) == len(candidates)
        filtered_dataset = filtered_dataset.add_column("candidates", candidates)
        processed_dataset = self.data_preprocess(filtered_dataset, train=False)
        dataloader = DataLoader(
            processed_dataset,
            batch_size=batch_size,
            sampler=SequentialSampler(processed_dataset),
            collate_fn=CollatorForExtend(self.tokenizer),
        )

        pbar  = tqdm(total=len(dataloader), desc='Evaluate')
        predictions = []
        for batch in dataloader:
            pbar.update()
            _, scores = self.model(**batch) # (2, batch_size, seq_length)
            probabilities = torch.softmax(scores, dim=-1) # (2, batch_size, seq_length)
            for i in range(batch['candidates_offsets'].size(0)):
                start_index, end_index = select_indices(
                    mode=self.config.mode,
                    possible_indices=batch["candidates_offsets"][i],
                    classification_probabilities=probabilities[:, i],
                )
                pred = batch['candidates_offsets'][i].tolist().index([start_index.item(), end_index.item()])
                predictions.append(pred)
        pbar.close()

        num_golds = len(filtered_dataset)
        num_corrects = 0
        for i, data in enumerate(filtered_dataset):
            candidates = data['candidates']
            pred = candidates[predictions[i]]
            if pred in data["label"]:
                num_corrects += 1
        metric = calculate_top1_accuracy(num_corrects, num_golds)
        return metric

    @torch.no_grad()
    def predict(self, sentence: str, spans: Optional[list[tuple[int, int]]] = None, num_candidates: int = 30) -> list[list[dict[str, Any]]]:
        if not spans:
            raise ValueError("Spans must be provided for ExtEnD prediction.")
        self.model.eval()
        self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        all_result = []
        candidates = self.retriever.predict(sentence, spans, top_k=num_candidates)
        for i, (b, e) in enumerate(spans):
            candidates_ids = [cand['id'] for cand in candidates[i]]
            processed_text = self.process_context(sentence, b, e, candidates_ids)
            _ = processed_text.pop("offset_mapping", None)
            encodings = self.tokenizer.pad([processed_text], return_tensors='pt')
            _, scores = self.model(**encodings)
            probabilities = torch.softmax(scores, dim=-1) # (2, batch_size, seq_length)
            start_index, end_index = select_indices(
                mode=self.config.mode,
                possible_indices=encodings["candidates_offsets"][0],
                classification_probabilities=probabilities[:, 0],
            )
            pred = encodings['candidates_offsets'][0].tolist().index([start_index.item(), end_index.item()])
            all_result.append([candidates[i][pred]])

        return all_result

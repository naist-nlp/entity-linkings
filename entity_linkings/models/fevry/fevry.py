from dataclasses import dataclass
from typing import Any, Iterator, Optional

import torch
from datasets import Dataset
from torch.utils.data import DataLoader, SequentialSampler
from tqdm.auto import tqdm
from transformers import (
    BatchEncoding,
    EvalPrediction,
    set_seed,
)

from entity_linkings.data_utils import (
    CollatorForReranking,
    truncate_around_mention,
)
from entity_linkings.dataset.utils import preprocess
from entity_linkings.trainer import EntityLinkingTrainer, TrainingArguments
from entity_linkings.utils import calculate_top1_accuracy

from ..base import EntityRetrieverBase, PipelineBase
from .span_classifier import SpanClassifier


class FEVRY(PipelineBase):

    @dataclass
    class Config(PipelineBase.Config):
        '''FEVRY configuration
        '''
        model_name_or_path: str = "google-bert/bert-base-uncased"

    def __init__(self, retriever: EntityRetrieverBase, config: Optional[Config] = None) -> None:
        super().__init__(retriever, config)
        self.model = SpanClassifier(self.config.model_name_or_path, len(self.dictionary))
        self.model.resize_token_embeddings(len(self.tokenizer))

    def compute_metrics(self, p: EvalPrediction) -> dict[str, float]:
        loss, (_, scores) = p.predictions
        preds = scores.argmax(axis=1).ravel()
        labels = p.label_ids.ravel()
        mask = labels != -100
        preds = preds[mask]
        labels = labels[mask]

        num_corrects = (preds == labels).sum().item()
        num_golds = mask.sum().item()
        accuracy = num_corrects / num_golds if num_golds > 0 else float("nan")
        return {"loss": loss.sum(), "accuracy": accuracy}

    def data_preprocess(self, dataset: Dataset) -> Dataset:
        def _preprocess_example(examples: Dataset) -> Iterator[BatchEncoding]:
            available_length = self.config.max_context_length - 2  # for [CLS] and [SEP]
            for text, entities in zip(examples["text"], examples["entities"]):
                for ent in entities:
                    if not ent["label"]:
                        continue
                    head = self.tokenizer.encode(text[: ent["start"]], add_special_tokens=False)
                    mention = self.tokenizer.encode(text[ent["start"]: ent["end"]], add_special_tokens=False)
                    tail = self.tokenizer.encode(text[ent["end"]:], add_special_tokens=False)

                    input_ids = head + mention + tail
                    input_ids, new_b, new_e = truncate_around_mention(input_ids, available_length, len(head), len(head) + len(mention))

                    encodings = self.tokenizer.prepare_for_model(
                        input_ids,
                        truncation=True,
                        max_length=self.config.max_context_length,
                        add_special_tokens=True
                    )
                    if encodings['input_ids'] == input_ids:
                        # No special tokens were added
                        encodings["input_ids"] = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]

                    assert len(encodings["input_ids"]) <= self.config.max_context_length
                    assert new_b < len(encodings["input_ids"])
                    assert new_e < len(encodings["input_ids"])
                    encodings["start_positions"] = new_b + 1  # +1 for [CLS]
                    encodings["end_positions"] = new_e + 1 # +1 for [CLS]
                    encodings["labels"] = ent["label"]
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
        train_dataset = self.data_preprocess(train_dataset)
        train_dataset = train_dataset.add_column("candidates", train_candidates)

        if eval_dataset is not None:
            eval_candidates = self.retriever.retrieve_candidates(
                eval_dataset,
                only_negative=True,
                top_k=num_candidates,
                batch_size=training_args.per_device_eval_batch_size
            )
            eval_dataset = self.data_preprocess(eval_dataset)
            eval_dataset = eval_dataset.add_column("candidates", eval_candidates)

        trainer = EntityLinkingTrainer(
            model=self.model,
            args=training_args,
            data_collator=CollatorForReranking(
                self.tokenizer,
                dictionary=self.dictionary,
                train=True,
            ),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
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
        self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        candidates = self.retriever.retrieve_candidates(
            dataset,
            only_negative=False,
            top_k=num_candidates,
            batch_size=batch_size
        )
        processed_dataset = self.data_preprocess(dataset)
        processed_dataset = processed_dataset.add_column("candidates", candidates)
        dataloader = DataLoader(
            processed_dataset,
            batch_size=batch_size,
            sampler=SequentialSampler(processed_dataset),
            collate_fn=CollatorForReranking(
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
            batch = batch.to("cuda" if torch.cuda.is_available() else "cpu")
            _, scores = self.model(**batch) # (batch_size, n_candidates)
            scores = scores.to('cpu')
            preds = scores.argmax(axis=1).ravel()  # (batch_size, )]
            labels = labels.ravel()
            mask = labels != -100
            preds = preds[mask]
            labels = labels[mask]
            num_corrects += (preds == labels).sum().item()
            num_golds += mask.sum().item()
        pbar.close()
        metric = calculate_top1_accuracy(num_corrects, num_golds)
        return metric

    @torch.no_grad()
    def predict(self, sentence: str, spans: Optional[list[tuple[int, int]]] = None, num_candidates: int = 30) -> list[list[dict[str, Any]]]:
        if not spans:
            raise ValueError("Spans must be provided for FEVRY prediction.")
        self.model.eval()
        self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        candidates = self.retriever.predict(sentence, spans, top_k=num_candidates)
        all_result = []
        available_length = self.config.max_context_length - 2  # for [CLS] and [SEP]
        for i, (b, e) in enumerate(spans):
            head = self.tokenizer.encode(sentence[: b], add_special_tokens=False)
            mention = self.tokenizer.encode(sentence[b: e], add_special_tokens=False)
            tail = self.tokenizer.encode(sentence[e:], add_special_tokens=False)

            input_ids = head + mention + tail
            input_ids, new_b, new_e = truncate_around_mention(input_ids, available_length, len(head), len(head) + len(mention))

            encodings = self.tokenizer.prepare_for_model(
                input_ids,
                truncation=True,
                max_length=self.config.max_context_length,
                add_special_tokens=True,
                return_tensors="pt",
            )
            if len(encodings['input_ids']) == input_ids:
                # No special tokens were added
                encodings['input_ids'] = torch.tensor([[self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]])

            encodings = self.tokenizer.pad([encodings])
            encodings["start_positions"] = torch.tensor([new_b + 1])  # +1 for [CLS]
            encodings["end_positions"] = torch.tensor([ new_e + 1])
            encodings["candidates_ids"] = torch.tensor([[self.dictionary.get_label_id(cand_id['id']) for cand_id in candidates[i]]])

            encodings = encodings.to("cuda" if torch.cuda.is_available() else "cpu")
            _, scores = self.model(**encodings)
            scores = scores.to('cpu')
            preds = scores.argmax(axis=1)  # (1,)
            predict = self.dictionary[preds[0].item()]
            all_result.append([predict])
        return all_result

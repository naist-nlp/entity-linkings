from dataclasses import dataclass
from typing import Any, Iterator, Optional

import torch
from datasets import Dataset
from torch.utils.data import DataLoader, SequentialSampler
from tqdm.auto import tqdm
from transformers import BatchEncoding, EvalPrediction, set_seed
from transformers.data.data_collator import pad_without_fast_tokenizer_warning

from entity_linkings.data_utils import (
    CollatorForCrossEncoder,
    EntityDictionary,
    cut_context_window,
)
from entity_linkings.dataset.utils import preprocess
from entity_linkings.trainer import EntityLinkingTrainer, TrainingArguments
from entity_linkings.utils import calculate_top1_accuracy

from ..base import EntityRetrieverBase, PipelineBase
from .crossencoder import CrossEncoder


class BLINK(PipelineBase):
    """ BLINK: Scalable Zero-shot Entity Linking with Dense Entity Retrieval (https://aclanthology.org/2020.emnlp-main.519/)
    """

    @dataclass
    class Config(PipelineBase.Config):
        """ BLINK configuration
        """
        model_name_or_path: str = "google-bert/bert-base-uncased"
        ent_start_token: str = "[START_ENT]"
        ent_end_token: str = "[END_ENT]"
        entity_token: str = "[ENT]"
        max_candidate_length: int = 50
        pooling: str = 'first'
        context_window_chars: int = 500

    def __init__(self, retriever: EntityRetrieverBase, config: Optional[Config] = None) -> None:
        super().__init__(retriever, config)
        special_tokens = [self.config.ent_start_token, self.config.ent_end_token, self.config.entity_token, self.config.nil_token]
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        self.model = CrossEncoder(self.config.model_name_or_path, pooling=self.config.pooling)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.dictionary = self.dictionary_preprocess(self.dictionary)

    def convert_mention_template(self, text: str, start: int, end: int) -> str:
        return text[:start] + self.config.ent_start_token + text[start:end] + self.config.ent_end_token + text[end:]

    def convert_entity_template(self, name: str, description: str) -> str:
        return name + self.config.entity_token + description

    def data_preprocess(self, dataset: Dataset) -> Dataset:
        def _preprocess_example(examples: Dataset) -> Iterator[BatchEncoding]:
            for text, entities in zip(examples["text"], examples["entities"]):
                for ent in entities:
                    if not ent["label"]:
                        continue
                    context, new_start, new_end = cut_context_window(text, ent["start"], ent["end"], self.config.context_window_chars)
                    context = self.convert_mention_template(context, new_start, new_end)
                    encodings = self.tokenizer(
                        context,
                        truncation=True,
                        max_length=self.config.max_context_length,
                        add_special_tokens=False,
                    )
                    encodings["labels"] = ent["label"]
                    yield encodings
        return preprocess(dataset, _preprocess_example)

    def dictionary_preprocess(self, dictionary: EntityDictionary) -> EntityDictionary:
        def preprocess_example(name: str, description: str) -> dict[str, list[int]]:
            text = self.convert_entity_template(name, description)
            encodings  = self.tokenizer.encode(
                text,
                padding=True,
                truncation=True,
                add_special_tokens=False,
                max_length=self.config.max_candidate_length,
            )
            return encodings
        dictionary.add_encoding(preprocess_example)
        return dictionary

    def compute_metrics(self, p: EvalPrediction) -> dict[str, float]:
        loss, (_, scores) = p.predictions

        labels = p.label_ids # (n_samples, n_candidates)
        preds = scores.argmax(axis=1)  # (batch_size, )
        labels = labels[range(preds.shape[0]), preds] # (n_samples, )
        num_corrects = labels.sum().item()
        num_golds = labels.shape[0]
        accuracy = num_corrects / num_golds if num_golds > 0 else float("nan")
        return {"loss": loss.sum(), "accuracy": accuracy}

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
            negative=True,
            top_k=num_candidates,
            batch_size=training_args.per_device_eval_batch_size,
        )
        train_dataset = self.data_preprocess(train_dataset)
        train_dataset = train_dataset.add_column("candidates", train_candidates)

        if eval_dataset is not None:
            eval_candidates = self.retriever.retrieve_candidates(
                eval_dataset,
                negative=True,
                top_k=self.config.num_candidates,
                batch_size=training_args.per_device_eval_batch_size,
            )
            eval_dataset = self.data_preprocess(eval_dataset)
            eval_dataset = eval_dataset.add_column("candidates", eval_candidates)

        trainer = EntityLinkingTrainer(
            model=self.model,
            args=training_args,
            data_collator=CollatorForCrossEncoder(
                self.tokenizer,
                dictionary=self.dictionary,
                train=True
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
            batch = batch.to("cuda" if torch.cuda.is_available() else "cpu")
            _, scores = self.model(**batch) # (batch_size, n_candidates)
            scores = scores.to('cpu')
            preds = scores.argmax(axis=1)  # (batch_size, )]
            labels = labels[range(preds.size(0)), preds]
            num_corrects = labels.sum().item()
            num_golds += labels.size(0)
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
        for i, (b, e) in enumerate(spans):
            context, new_b, new_e = cut_context_window(sentence, b, e, self.config.context_window_chars)
            context = self.convert_mention_template(context, new_b, new_e)
            context_tokens = self.tokenizer.encode(
                context,
                add_special_tokens=False, truncation=True,max_length=self.config.max_context_length
            )
            candidates_encodings = []
            for cand_id in candidates[i]:
                candidate = self.dictionary(cand_id['id'])
                candidate_tokens = self.tokenizer.encode(
                    self.convert_entity_template(candidate["name"], candidate["description"]),
                    add_special_tokens=False, truncation=True, max_length=self.config.max_candidate_length
                )
                candidate_encodings = self.tokenizer.prepare_for_model(
                    context_tokens + candidate_tokens,
                    add_special_tokens=True,
                    padding=True
                )
                if len(candidate_encodings['input_ids']) == context_tokens + candidate_tokens:
                    # No special tokens were added
                    candidate_encodings['input_ids'] = torch.tensor([[self.tokenizer.cls_token_id] + context_tokens + candidate_tokens + [self.tokenizer.sep_token_id]])
                candidates_encodings.append(candidate_encodings)

            encodings= pad_without_fast_tokenizer_warning(
                self.tokenizer,
                candidates_encodings,
                padding=True,
                return_tensors="pt"
            ) # (n_candidates, ...)
            encodings = encodings.to("cuda" if torch.cuda.is_available() else "cpu")
            scores = self.model.score(**encodings) # (n_candidates, )
            scores = scores.to('cpu')
            preds = scores.argsort(descending=True)  # (n_candidates, )
            predict = [candidates[i][ind] for ind in preds]
            all_result.append(predict)
        return all_result

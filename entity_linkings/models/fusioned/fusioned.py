from dataclasses import dataclass
from logging import getLogger
from typing import Any, Iterator, Optional

import torch
from datasets import Dataset
from torch.utils.data import DataLoader, SequentialSampler
from tqdm.auto import tqdm
from transformers import AutoTokenizer, BatchEncoding, set_seed
from transformers.data.data_collator import pad_without_fast_tokenizer_warning

from entity_linkings.data_utils import (
    CollatorForFusioned,
    EntityDictionary,
    truncate_around_mention,
)
from entity_linkings.dataset.utils import preprocess
from entity_linkings.trainer import EntityLinkingTrainer, TrainingArguments
from entity_linkings.utils import calculate_top1_accuracy

from ..base import EntityRetrieverBase, PipelineBase
from .reader import FusionELReader

logger = getLogger(__name__)
logger.setLevel("INFO")


class FUSIONED(PipelineBase):
    """ Entity Disambiguation via Fusion Entity Decoding
    """

    @dataclass
    class Config(PipelineBase.Config):
        model_name_or_path: str = "google/flan-t5-base"
        max_context_length: int = 250
        max_candidate_length: int = 140
        max_generation_length: int = 100
        document_token: str = "<extra_id_0>"
        passage_token: str = "<extra_id_1>"
        title_token: str = "<extra_id_2>"
        description_token: str = "<extra_id_3>"
        entity_token: str = "<extra_id_4>"
        mention_token: str = "<extra_id_5>"
        ent_start_token: str = "<extra_id_6>"
        ent_end_token: str = "<extra_id_7>"

    def __init__(self, retriever: EntityRetrieverBase, config: Optional[Config] = None) -> None:
        super().__init__(retriever, config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name_or_path)
        self.model = FusionELReader(self.config.model_name_or_path)
        self.dictionary = self.dictionary_preprocess(self.dictionary)

    def process_context(self, text: str, start: int, end: int) -> BatchEncoding:
        available_length = self.config.max_context_length - 1 # for </s> token
        head = self.tokenizer.encode(text[: start] + self.config.ent_start_token, add_special_tokens=False)
        mention = self.tokenizer.encode(text[start: end], add_special_tokens=False)
        tail = self.tokenizer.encode(self.config.ent_end_token + text[end:], add_special_tokens=False)

        input_ids = head + mention + tail
        input_ids, mention_start, mention_end = truncate_around_mention(input_ids, available_length, len(head), len(head) + len(mention))
        assert len(input_ids) <= available_length

        encodings = self.tokenizer.prepare_for_model(
            input_ids,
            truncation=True,
            max_length=self.config.max_context_length,
            add_special_tokens=False,
        )
        assert len(encodings["input_ids"]) <= self.config.max_context_length
        assert encodings["input_ids"][mention_start: mention_end] == mention
        return encodings

    def data_preprocess(self, dataset: Dataset) -> Dataset:
        def _preprocess_example(example: dict) -> Iterator[BatchEncoding]:
            for text, entities in zip(example["text"], example["entities"]):
                for ent in entities:
                    if not ent["label"]:
                        continue
                    encodings = self.process_context(text, ent["start"], ent["end"])
                    encodings["labels"] = ent["label"]
                    yield encodings
        return preprocess(dataset, _preprocess_example)

    def dictionary_preprocess(self, dictionary: EntityDictionary) -> EntityDictionary:
        def preprocess_example(name: str, description: str) -> dict[str, list[int]]:
            text = self.config.title_token + name + self.config.description_token + description
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
            data_collator=CollatorForFusioned(
                self.tokenizer, dictionary=self.dictionary, train=True
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
        self.model.eval()
        candidates = self.retriever.retrieve_candidates(dataset, top_k=num_candidates, only_negative=False, batch_size=batch_size)
        processed_dataset = self.data_preprocess(dataset)
        processed_dataset = processed_dataset.add_column("candidates", candidates)

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
            batch = batch.to("cuda" if torch.cuda.is_available() else "cpu")
            generated_ids = self.model.generate(**batch, max_generation_length=self.config.max_generation_length)
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
    def predict(self, sentence: str, spans: Optional[list[tuple[int, int]]] = None, num_candidates: int = 30) -> list[list[dict[str, Any]]]:
        if not spans:
            raise ValueError("Spans must be provided for FUSIONED prediction.")
        self.model.eval()
        self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        all_result = []
        candidates = self.retriever.predict(sentence, spans, top_k=num_candidates)
        for i, (b, e) in enumerate(spans):
            encoding = self.process_context(sentence, b, e)
            encodings = []
            for cand in candidates[i]:
                encodings.append(self.tokenizer.prepare_for_model(
                    encoding['input_ids'] + self.dictionary(cand['id'])['encoding'],
                    truncation=True,
                    add_special_tokens=True,
                ))

            batch = pad_without_fast_tokenizer_warning(
                self.tokenizer,
                encodings,
                padding=True,
                return_tensors="pt",
            )
            for k, v in batch.items():
                batch[k] = v.unsqueeze(0)  # add batch dimension

            batch = batch.to("cuda" if torch.cuda.is_available() else "cpu")
            generated_ids = self.model.generate(**batch, max_generation_length=self.config.max_generation_length)
            generated_ids = generated_ids.to('cpu')
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            candidate_titles = [cand['prediction'] for cand in candidates[i]]
            if generated_text not in candidate_titles:
                logger.warning(f"FUSIONED could not find a valid entity for the mention: {sentence[spans[i][0]:spans[i][1]]}")
                all_result.append([])
                continue
            pred = candidate_titles.index(generated_text)
            all_result.append([candidates[i][pred]])
        return all_result

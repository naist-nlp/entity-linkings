import logging
from typing import Any, Callable, Iterator, Optional

from datasets import Dataset
from transformers import BatchEncoding, PreTrainedTokenizer, TrainingArguments

from .entity_dictionary import EntityDictionary
from .utils import cut_context_window, truncate_around_mention

logger = logging.getLogger(__name__)


def preprocess(
        dataset: Dataset,
        processing_func: Callable[[Dataset], Iterator[Any]],
        training_arguments: Optional[TrainingArguments] =None,
        desc: Optional[str] = None
        ) -> dict[str, Dataset]:
    def _preprocess(documents: Dataset) -> Any:
        features = [_ for _ in processing_func(documents)]
        outputs = {}
        for k in list(features[0].keys()):
            outputs[k] = [f[k] for f in features]
        return outputs

    if training_arguments:
        with training_arguments.main_process_first(desc="dataset map pre-processing"):
            column_names = dataset.column_names
            splits = dataset.map(_preprocess, batched=True, remove_columns=column_names, desc=desc)
    else:
        column_names = dataset.column_names
        splits = dataset.map(_preprocess, batched=True, remove_columns=column_names, desc=desc)

    return splits


class Preprocessor:
    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            ent_start_token: str = "[START_ENT]",
            ent_end_token: str = "[END_ENT]",
            entity_token: str = "[ENT]",
            max_context_length: int = 250,
            max_candidate_length: int = 100,
            context_window_chars: int = 1000,
        ) -> None:
        self.tokenizer = tokenizer
        self.ent_start_token = ent_start_token
        self.ent_end_token = ent_end_token
        self.entity_token = entity_token
        self.max_context_length = max_context_length
        self.max_candidate_length = max_candidate_length
        self.context_window_chars = context_window_chars

        self.start_marker_id = self.tokenizer.convert_tokens_to_ids(self.ent_start_token)
        self.end_marker_id = self.tokenizer.convert_tokens_to_ids(self.ent_end_token)

        # Prepare prefix and suffix ids for special tokens
        if self.tokenizer.cls_token:
            self.prefix_ids = [self.tokenizer.cls_token_id]
        else:
            dummy_id = -1
            with_special_tokens = self.tokenizer.build_inputs_with_special_tokens([dummy_id])
            dummy_idx = with_special_tokens.index(dummy_id)
            self.prefix_ids = with_special_tokens[:dummy_idx]

        if self.tokenizer.sep_token:
            self.suffix_ids = [self.tokenizer.sep_token_id]
        else:
            dummy_id = -1
            with_special_tokens = self.tokenizer.build_inputs_with_special_tokens([dummy_id])
            dummy_idx = with_special_tokens.index(dummy_id)
            self.suffix_ids = with_special_tokens[dummy_idx + 1 :]

        self.offset_correction = len(self.prefix_ids)
        self.max_context_length = self.max_context_length - len(self.prefix_ids) - len(self.suffix_ids)

    def process_context(self, text: str, start: int, end: int, candidate_ids: Optional[list[str]] = None, labels: Optional[list[str]] = None) -> BatchEncoding:
        context, new_start, new_end = cut_context_window(text, start, end, self.context_window_chars)
        marked_text = context[:new_start] + self.ent_start_token + context[new_start:new_end] + self.ent_end_token + context[new_end:]
        input_ids = self.tokenizer.encode(marked_text, add_special_tokens=False, truncation=False)

        start_marker_idx = input_ids.index(self.start_marker_id)
        end_marker_idx = input_ids.index(self.end_marker_id)

        truncated_input_ids, mention_start, mention_end = truncate_around_mention(input_ids, self.max_context_length, start_marker_idx, end_marker_idx + 1)
        final_input_ids = self.prefix_ids + truncated_input_ids + self.suffix_ids
        final_start = mention_start + self.offset_correction
        final_end = mention_end + self.offset_correction

        encodings = BatchEncoding({
            "input_ids": final_input_ids,
            "attention_mask": [1] * len(final_input_ids),
            "start_positions": final_start,
            "end_positions": final_end,
        })
        if labels is not None:
            encodings["labels"] = labels
        if candidate_ids is not None:
            encodings["candidates"] = candidate_ids
        return encodings

    def process_candidate(self, name: str, description: str) -> BatchEncoding:
        marked_text = name + self.entity_token + description
        encodings = self.tokenizer.encode(
            marked_text,
            max_length=self.max_candidate_length,
            add_special_tokens=False
        )
        return encodings

    def data_flatten(self, dataset: Dataset) -> Dataset:
        def _preprocess_example(examples: Dataset) -> Iterator[dict]:
            for id_, text, entities in zip(examples["id"], examples["text"], examples["entities"]):
                for ent in entities:
                    if not ent["label"]:
                        continue
                    yield {"id": id_, "text": text, "entity": ent}
        return preprocess(dataset, _preprocess_example, desc="Filtering")

    def dataset_preprocess(self, dataset: Dataset, candidates_ids: Optional[list[list[str]]] = None) -> Dataset:
        def _preprocess_example(examples: Dataset) -> Iterator[BatchEncoding]:
            for i, (text, entity) in enumerate(zip(examples["text"], examples["entity"])):
                candidate_ids = examples["candidates"][i] if "candidates" in examples else None
                encodings = self.process_context(text, entity['start'], entity['end'], candidate_ids, entity['label'] if 'label' in entity else None)
                yield encodings
        flatten_dataset = self.data_flatten(dataset)
        if candidates_ids is not None:
            assert len(flatten_dataset) == len(candidates_ids)
            flatten_dataset = flatten_dataset.add_column("candidates", candidates_ids)
        processed_dataset = preprocess(flatten_dataset, _preprocess_example, desc="Preprocessing")
        return processed_dataset

    def dictionary_preprocess(self, dictionary: EntityDictionary) -> EntityDictionary:
        def preprocess_example(name: str, description: str) -> BatchEncoding:
            encodings  = self.process_candidate(name, description)
            return encodings
        dictionary.add_encoding(preprocess_example)
        return dictionary

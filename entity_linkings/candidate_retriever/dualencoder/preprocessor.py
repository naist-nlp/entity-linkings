from typing import Iterator, Optional

from datasets import Dataset
from transformers import BatchEncoding

from entity_linkings.data_utils import Preprocessor, preprocess
from entity_linkings.data_utils.utils import cut_context_window, truncate_around_mention


class DualEncoderPreprocessor(Preprocessor):
    def process_candidate(self, name: str, description: str) -> BatchEncoding:
        marked_text = name + self.entity_token + description
        encodings  = self.tokenizer(
            marked_text,
            padding=True,
            truncation=True,
            max_length=self.max_candidate_length,
        )
        return encodings

    def _process_context(self, text: str, start: int, end: int) -> str:
        context, new_start, new_end = cut_context_window(text, start, end, self.context_window_chars)
        marked_text = context[:new_start] + self.ent_start_token + context[new_start:new_end] + self.ent_end_token + context[new_end:]
        return marked_text

    def process_context(self, text: str, start: int, end: int, candidate_ids: Optional[list[str]] = None, labels: Optional[list[str]] = None) -> BatchEncoding:
        marked_text = self._process_context(text, start, end)
        input_ids = self.tokenizer.encode(marked_text, add_special_tokens=False, truncation=False)

        start_marker_idx = input_ids.index(self.start_marker_id)
        end_marker_idx = input_ids.index(self.end_marker_id)

        truncated_input_ids, _, _ = truncate_around_mention(input_ids, self.max_context_length, start_marker_idx, end_marker_idx + 1)
        final_input_ids = self.prefix_ids + truncated_input_ids + self.suffix_ids

        encodings = BatchEncoding({
            "input_ids": final_input_ids,
            "attention_mask": [1] * len(final_input_ids),
            "token_type_ids": [0] * len(final_input_ids),
        })
        if labels is not None:
            encodings["labels"] = labels
        if candidate_ids is not None:
            encodings["candidates"] = candidate_ids
        return encodings

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

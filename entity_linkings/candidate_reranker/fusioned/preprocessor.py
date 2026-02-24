from typing import Any, Optional

from transformers import BatchEncoding

from entity_linkings.data_utils import (
    EntityDictionary,
    Preprocessor,
    cut_context_window,
    truncate_around_mention,
)


class FusionedPreprocessor(Preprocessor):
    def __init__(self, title_token: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.title_token = title_token

    def dictionary_preprocess(self, dictionary: EntityDictionary) -> EntityDictionary:
        def preprocess_example(name: str, description: str) -> dict[str, list[int]]:
            text = self.title_token + name + self.entity_token + description
            encodings  = self.tokenizer.encode(
                text,
                padding=True,
                truncation=True,
                add_special_tokens=False,
                max_length=self.max_candidate_length,
            )
            return encodings
        dictionary.add_encoding(preprocess_example)
        return dictionary

    def process_context(self, text: str, start: int, end: int, candidate_ids: Optional[list[str]] = None, labels: Optional[list[str]] = None) -> BatchEncoding:
        context, new_start, new_end = cut_context_window(text, start, end, self.context_window_chars)
        marked_text = context[:new_start] + self.ent_start_token + context[new_start:new_end] + self.ent_end_token + context[new_end:]
        input_ids = self.tokenizer.encode(marked_text, add_special_tokens=False, truncation=False)

        start_marker_idx = input_ids.index(self.start_marker_id)
        end_marker_idx = input_ids.index(self.end_marker_id)

        truncated_input_ids, _, _ = truncate_around_mention(input_ids, self.max_context_length, start_marker_idx, end_marker_idx + 1)
        encodings = BatchEncoding({
            "input_ids": truncated_input_ids,
            "attention_mask": [1] * len(truncated_input_ids),
        })
        if labels is not None:
            encodings["labels"] = labels
        if candidate_ids is not None:
            encodings["candidates"] = candidate_ids
        return encodings

import random
from typing import Any, Iterator, Optional

from datasets import Dataset
from transformers import BatchEncoding

from entity_linkings.data_utils import (
    EntityDictionary,
    Preprocessor,
    cut_context_window,
    preprocess,
    truncate_around_mention,
)


def process_candidates(
        candidate_titles: list[str],
        gold_titles: list[str],
        separator: str = '*'
    )-> tuple[str, list[int], list[int], list[tuple[int, int]]]:
    context = ""
    candidates_offsets = []
    answer_starts, answer_ends = [], []
    for candidate_title in candidate_titles:
        candidate_start = len(context)
        candidate_end = candidate_start + len(candidate_title)
        candidates_offsets.append( (candidate_start, candidate_end) )
        if candidate_title in gold_titles:
            answer_starts.append(candidate_start)
            answer_ends.append(candidate_end)
        context += candidate_title + f" {separator} "
    return context, answer_starts, answer_ends, candidates_offsets


def compute_char_to_tokens(context: str, context_mask: list[bool], offsets_map: list[tuple[int, int]]) -> dict[int, int]:
    char2token = {}
    first = True
    for _t_idx, (m, cp) in enumerate(zip(context_mask, offsets_map)):
        if m:
            while (
                offsets_map[_t_idx][0] < offsets_map[_t_idx][1]
                and context[offsets_map[_t_idx][0]] == " "
            ):
                offsets_map[_t_idx][0] += 1

            # add prefix space seems to be bugged on some tokenizers
            if first:
                first = False
                if cp[0] != 0 and context[cp[0] - 1] != " ":
                    offsets_map[_t_idx][0] -= 1
                    cp = (cp[0] - 1, cp[1])
            if cp[0] == cp[1]:
                assert context[cp[0] - 1] == " ", f"Token {_t_idx} found to occur at char span ({cp[0]}, {cp[1]}), which is impossible"
            for c in range(*cp):
                char2token[c] = _t_idx
    return char2token


class ExtendPreprocessor(Preprocessor):
    def __init__(self, dictionary: EntityDictionary, candidate_separator: str = "*", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.dictionary = dictionary
        self.candidate_separator = candidate_separator

    def process_context(
            self, text: str, start: int, end: int,
            candidate_ids: Optional[list[str]] = None, labels: Optional[list[str]] = None,
            train: bool = False
        ) -> BatchEncoding:
        if candidate_ids is None:
            raise ValueError("candidate_ids must be provided for ExtendPreprocessor.")

        context, new_start, new_end = cut_context_window(text, start, end, self.context_window_chars)
        marked_text = context[:new_start] + self.ent_start_token + context[new_start:new_end] + self.ent_end_token + context[new_end:]
        input_ids = self.tokenizer.encode(marked_text, add_special_tokens=False, truncation=False)

        start_marker_idx = input_ids.index(self.start_marker_id)
        end_marker_idx = input_ids.index(self.end_marker_id)

        trunscated_input_ids, _, _ = truncate_around_mention(input_ids, self.max_context_length, start_marker_idx, end_marker_idx + 1)

        if labels is not None:
            gold_titles = [self.dictionary(labels[0])['name']]
            if train:
                candidate_ids = candidate_ids[:-1] + [labels[0]]
                random.shuffle(candidate_ids)
        else:
            gold_titles = []

        candidate_titles = [self.dictionary(cand)['name'] for cand in candidate_ids]
        candidate_context, answer_starts, answer_ends, candidates_offsets = process_candidates(
            candidate_titles, gold_titles, separator=self.candidate_separator
        )

        encodings = self.tokenizer(
            self.tokenizer.decode(trunscated_input_ids), candidate_context,
            return_offsets_mapping=True,
        )
        char2token = compute_char_to_tokens(
            candidate_context,
            [p == 1 for p in encodings.sequence_ids()],
            encodings["offset_mapping"]
        )

        encodings["candidates"] = candidate_ids
        encodings["candidates_offsets"] = [
            (char2token[si], char2token[ei - 1] + 1)
            for si, ei in candidates_offsets
        ]
        if labels is not None:
            if train:
                encodings["start_positions"] = [char2token[ans_start] for ans_start in answer_starts]
                encodings["end_positions"] = [char2token[ans_end - 1] + 1 for ans_end in answer_ends]
            encodings["labels"] = labels

        return encodings

    def dataset_preprocess(self, dataset: Dataset, candidates_ids: Optional[list[list[str]]] = None, train: bool = False) -> Dataset:
        def _preprocess_example(examples: Dataset) -> Iterator[BatchEncoding]:
            for i, (text, entity) in enumerate(zip(examples["text"], examples["entity"])):
                candidate_ids = examples["candidates"][i] if "candidates" in examples else None
                encodings = self.process_context(
                    text, entity['start'], entity['end'], candidate_ids,
                    entity['label'] if 'label' in entity else None, train=train
                )
                yield encodings
        flatten_dataset = self.data_flatten(dataset)
        if candidates_ids is not None:
            assert len(flatten_dataset) == len(candidates_ids)
            flatten_dataset = flatten_dataset.add_column("candidates", candidates_ids)
        processed_dataset = preprocess(flatten_dataset, _preprocess_example, desc="Preprocessing")
        return processed_dataset

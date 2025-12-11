from importlib.resources import files

import assets as test_data
from entity_linkings import load_dataset, load_dictionary

from .utils import (
    cut_context_window,
    filter_nil_entities,
    sample_range_excluding,
    truncate_around_mention,
)

dataset_path = str(files(test_data).joinpath("dataset_toy.jsonl"))
dictionary_path = str(files(test_data).joinpath("dictionary_toy.jsonl"))
dictionary = load_dictionary(dictionary_path)
dataset = load_dataset(data_files={"test": dataset_path})['test']

def test_filter_nil_entities() -> None:
    filtered_dataset = filter_nil_entities(dataset, dictionary)
    for example in filtered_dataset:
        for entity in example["entities"]:
            for label in entity["label"]:
                assert dictionary(label) != dictionary.nil_id
    assert len(filtered_dataset) == len(dataset)
    assert len(filtered_dataset) == 8


def test_sample_range_excluding() -> None:
    n, k = 10, 3
    excluding = [2, 5, 7]
    sampled = sample_range_excluding(n, k, excluding)
    assert len(sampled) == k
    for s in sampled:
        assert s not in excluding
        assert 0 <= s < n

def test_cut_context_window() -> None:
    text = "This is a sample text for testing the cut_context_window function."
    start, end = 10, 16  # "sample"
    context_window_chars = 30

    context, new_start, new_end = cut_context_window(text, start, end, context_window_chars)

    assert len(context) <= context_window_chars
    assert context[new_start:new_end] == "sample"

    # Test edge case where mention is at the start
    text_start = "sample text for testing the cut_context_window function."
    start_start, end_start = 0, 6  # "sample"
    context_start, new_start_start, new_end_start = cut_context_window(text_start, start_start, end_start, context_window_chars)
    assert context_start[new_start_start:new_end_start] == "sample"

    # Test edge case where mention is at the end
    text_end = "This is a sample text for testing the cut_context_window function sample"
    start_end, end_end = len(text_end) - 6, len(text_end)  # "sample"
    context_end, new_start_end, new_end_end = cut_context_window(text_end, start_end, end_end, context_window_chars)
    assert context_end[new_start_end:new_end_end] == "sample"


def test_truncate_around_mention() -> None:
    tokens_ids = list(range(100))
    max_token_length = 50
    ent_start_idx = 45
    ent_end_idx = 55

    truncated_ids, new_start, new_end = truncate_around_mention(
        tokens_ids, max_token_length, ent_start_idx, ent_end_idx
    )

    assert len(truncated_ids) <= max_token_length
    assert new_start >= 0 and new_end <= len(truncated_ids)
    assert new_end - new_start == ent_end_idx - ent_start_idx
    assert truncated_ids[new_start:new_end] == tokens_ids[ent_start_idx:ent_end_idx]

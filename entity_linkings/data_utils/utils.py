import bisect
import random
from typing import Any

from datasets import Dataset

from .entity_dictionary import EntityDictionary


def filter_nil_entities(dataset: Dataset, dictionary: EntityDictionary) -> Dataset:
    def filter_fn(example: dict[str, Any]) -> dict[str, Any]:
        entities = []
        for entity in example["entities"]:
            labels = [label for label in entity["label"] if dictionary(label) != dictionary.nil_id]
            if labels:
                entities.append({"start": entity["start"], "end": entity["end"], "label": labels})
        example.update({"entities": entities})
        return example
    return dataset.map(filter_fn)


def sample_range_excluding(n: int, k: int, excluding: list[int]) -> list[int]:
    skips = [j - i for i, j in enumerate(sorted(excluding))]
    s = random.sample(range(n - len(skips)), k)
    return [i + bisect.bisect_right(skips, i) for i in s]


def cut_context_window(text: str, start: int, end: int, context_window_chars: int) -> tuple[str, int, int]:
    mention_center = (start + end) // 2
    half_window = context_window_chars // 2

    window_start = max(0, mention_center - half_window)
    window_end = min(len(text), window_start + context_window_chars)

    if window_end - window_start < context_window_chars:
        window_start = max(0, window_end - context_window_chars)

    context = text[window_start: window_end]
    new_start = start - window_start
    new_end = end - window_start
    if new_start < 0 or new_end > len(context):
        raise ValueError("Adjusted mention span is out of context bounds.")
    return context, new_start, new_end


def truncate_around_mention(
    tokens_ids: list[int],
    max_token_length: int,
    ent_start_idx: int,
    ent_end_idx: int
) -> tuple[list[int], int, int]:
    # Compute window around mention
    mention_center = (ent_start_idx + ent_end_idx) // 2
    half_window = max_token_length // 2

    left_context_length = mention_center
    right_context_length = len(tokens_ids) - mention_center

    if left_context_length >= half_window and right_context_length >= half_window:
        # Both sides have enough space → symmetric window
        left = mention_center - half_window
        right = mention_center + half_window
    elif left_context_length < half_window:
        # Not enough on the left → start from 0, extend right
        left = 0
        right = min(len(tokens_ids), max_token_length)
    else:  # right_space < half_window
        # Not enough on the right → end at len(text_tokens), extend left
        right = len(tokens_ids)
        left = max(0, right - max_token_length)
    truncated_text_tokens = tokens_ids[left:right]

    new_ent_start_idx = max(0, ent_start_idx - left)
    new_ent_end_idx = min(len(truncated_text_tokens), ent_end_idx - left)

    return truncated_text_tokens, new_ent_start_idx, new_ent_end_idx

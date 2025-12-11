import pytest
import torch
from transformers import AutoTokenizer

from .utils import (
    calculate_joint_probabilities,
    compute_char_to_tokens,
    process_candidates,
    select_indices,
)

MODELS = [
    "google-bert/bert-base-uncased",
    "allenai/longformer-base-4096",
]


@pytest.mark.parametrize("gold_titles", [["Meta"], []])
def test_process_candidates(gold_titles: list[str]) -> None:
    candidate_titles = ["Apple", "Meta", "Amazon"]
    gold_titles = ["Meta", "Google"]
    context, answer_starts, answer_ends, candidates_offsets = process_candidates(
        candidate_titles,
        gold_titles,
        separator='*'
    )
    assert context == "Apple * Meta * Amazon * "
    assert answer_starts == [8] if gold_titles else []
    assert answer_ends == [12] if gold_titles else []
    assert candidates_offsets == [(0, 5), (8, 12), (15, 21)]


@pytest.mark.parametrize("model_name", MODELS)
def test_compute_char_to_tokens(model_name: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'additional_special_tokens': ['[START_ENT]', '[END_ENT]']})
    assert tokenizer.is_fast

    text = "Steve Jobs was found [START_ENT]Apple[END_ENT]."
    candidate_text = "Apple * Meta * Amazon * Google *"

    encodings = tokenizer(
        text,
        candidate_text,
        return_offsets_mapping=True,
    )

    char2token = compute_char_to_tokens(
        candidate_text,
        [p == 1 for p in encodings.sequence_ids()],
        encodings['offset_mapping']
    )
    assert isinstance(char2token, dict)


def test_calculate_joint_probabilities() -> None:
    classification_probabilities = torch.tensor([
        [0.1, 0.3, 0.6],
        [0.3, 0.2, 0.5],
    ])
    possible_indices = [(0, 2), (1, 1), (2, 0)]

    max_prod_probs = calculate_joint_probabilities(
        "max-prod",
        possible_indices,
        classification_probabilities
    )
    assert max_prod_probs == [
        torch.tensor(0.1) * torch.tensor(0.5),
        torch.tensor(0.3) * torch.tensor(0.2),
        torch.tensor(0.6) * torch.tensor(0.3),
    ]

    max_start_probs = calculate_joint_probabilities(
        "max-start",
        possible_indices,
        classification_probabilities
    )
    assert max_start_probs == [torch.tensor(0.1), torch.tensor(0.3), torch.tensor(0.6)]


    max_end_probs = calculate_joint_probabilities(
        "max-end",
        possible_indices,
        classification_probabilities
    )
    assert max_end_probs == [torch.tensor(0.5), torch.tensor(0.2), torch.tensor(0.3)]

    max_end_probs = calculate_joint_probabilities(
        "max",
        possible_indices,
        classification_probabilities
    )
    assert max_end_probs == [torch.tensor(0.5), torch.tensor(0.3), torch.tensor(0.6)]


def test_select_indices() -> None:
    classification_probabilities = torch.tensor([
        [0.1, 0.3, 0.6],
        [0.3, 0.2, 0.5],
    ])
    possible_indices = [(0, 2), (1, 1), (2, 0)]
    start_idx, end_idx = select_indices(
        "max-prod",
        possible_indices,
        classification_probabilities
    )
    assert (start_idx, end_idx) == (2, 0)

    start_idx, end_idx = select_indices(
        "max-start",
        possible_indices,
        classification_probabilities
    )
    assert (start_idx, end_idx) == (2, 0)

    start_idx, end_idx = select_indices(
        "max-end",
        possible_indices,
        classification_probabilities
    )
    assert (start_idx, end_idx) == (0, 2)

    start_idx, end_idx = select_indices(
        "max",
        possible_indices,
        classification_probabilities
    )
    assert (start_idx, end_idx) == (2, 0)

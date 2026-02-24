import torch

from .utils import (
    calculate_joint_probabilities,
    select_indices,
)


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

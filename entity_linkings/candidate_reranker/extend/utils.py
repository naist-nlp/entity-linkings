
from typing import Literal

import numpy as np
import torch
from transformers import EvalPrediction


def compute_metrics(p: EvalPrediction) -> dict[str, float]:
    start_logits, end_logits = p.predictions
    start_positions, end_positions = p.label_ids # (2, batch_size, n_samples)
    start_positions = start_positions.ravel() # (n_samples, )
    end_positions = end_positions.ravel() # (n_samples, )

    start_predictions = start_logits.argmax(axis=-1).ravel() # (n_samples, )
    end_predictions = end_logits.argmax(axis=-1).ravel() # (n_samples, )
    correct_full_predictions = np.logical_and(
        np.equal(start_predictions, start_positions),
        np.equal(end_predictions, end_positions),
    )
    accuracy = correct_full_predictions.mean().item()
    return {"accuracy": accuracy}


def calculate_joint_probabilities(
    mode: Literal["max-prod", "max-end", "max-start", "max"],
    possible_indices: list[tuple[int, int]],
    classification_probabilities: torch.Tensor,
) -> list[float]:
    def max_prod(x: tuple[int, int]) -> torch.Tensor:
        prob =  classification_probabilities[0][x[0]] * classification_probabilities[1][x[1]]
        return prob

    def max_start(x: tuple[int, int]) -> torch.Tensor:
        prob = classification_probabilities[0][x[0]]
        return prob

    def max_end(x: tuple[int, int]) -> torch.Tensor:
        prob = classification_probabilities[1][x[1]]
        return prob

    def max_func(x: tuple[int, int]) -> torch.Tensor:
        prob =  max(
            classification_probabilities[0][x[0]],
            classification_probabilities[1][x[1]],
        )
        return prob

    if mode == "max-prod":
        selector = max_prod
    elif mode == "max-end":
        selector = max_end
    elif mode == "max-start":
        selector = max_start
    elif mode == "max":
        selector = max_func
    else:
        raise NotImplementedError

    return [selector(x).item() for x in possible_indices]


def select_indices(
    mode: Literal["max-prod", "max-end", "max-start", "max"],
    possible_indices: list[tuple[int, int]],
    classification_probabilities: torch.Tensor,
) -> tuple[int, int]:

    prob_indices = calculate_joint_probabilities(
        mode=mode,
        possible_indices=possible_indices,
        classification_probabilities=classification_probabilities,
    )
    max_indice = prob_indices.index(max(prob_indices))
    return possible_indices[max_indice]

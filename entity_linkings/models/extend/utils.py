
from typing import Literal

import torch


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

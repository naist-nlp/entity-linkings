import torch

from .utils import first_token_pooler, last_token_pooler


def _hidden(batch_size: int = 3, seq_length: int = 5, hidden_size: int = 4) -> torch.Tensor:
    return torch.arange(batch_size * seq_length * hidden_size, dtype=torch.float).reshape(
        batch_size, seq_length, hidden_size
    )


def test_poolers_keep_one_row_per_sequence_under_left_padding() -> None:
    # first_token_pooler indexed with the lengths alone, which spreads the batch over a
    # second dimension instead of picking one position per row.
    hidden = _hidden()
    left_padded = torch.tensor([[0, 0, 1, 1, 1], [0, 0, 0, 1, 1], [0, 1, 1, 1, 1]])

    assert first_token_pooler(hidden, left_padded).size() == (3, 4)
    assert last_token_pooler(hidden, left_padded).size() == (3, 4)


def test_poolers_keep_one_row_per_sequence_under_right_padding() -> None:
    hidden = _hidden()
    right_padded = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 1, 0]])

    assert first_token_pooler(hidden, right_padded).size() == (3, 4)
    assert last_token_pooler(hidden, right_padded).size() == (3, 4)


def test_first_token_pooler_picks_the_first_real_token() -> None:
    hidden = _hidden(batch_size=1, seq_length=4, hidden_size=2)
    left_padded = torch.tensor([[0, 0, 1, 1]])

    # Row 0 is [[0,1],[2,3],[4,5],[6,7]]; the first token that is not padding is at 2.
    assert torch.equal(first_token_pooler(hidden, left_padded)[0], torch.tensor([4.0, 5.0]))

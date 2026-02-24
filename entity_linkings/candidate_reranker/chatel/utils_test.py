from .utils import process_multi_choice_prompt


def test_process_multi_choice_prompt() -> None:
    candidates = ["0001", "0002", "0003"]

    # Test case 1: Index selection
    result = process_multi_choice_prompt("(1).", [candidates[0]])
    assert result == 0  # Indexing starts from 0

    result = process_multi_choice_prompt("(2).", candidates)
    assert result == 1  # Indexing starts from 1

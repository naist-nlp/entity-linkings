import pytest
import torch

from .reader import FusionELReader

MODELS = ["google/flan-t5-small"]


def mock_input(
        batch_size: int = 2,
        num_candidates: int = 3,
        seq_length: int = 10,
        vocab_size: int = 100
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    import torch
    input_ids = torch.randint(0, vocab_size, (batch_size, num_candidates, seq_length))
    attention_mask = torch.ones((batch_size, num_candidates, seq_length), dtype=torch.long)
    labels = torch.randint(0, vocab_size, (batch_size, seq_length))
    return input_ids, attention_mask, labels


class TestFusionELReader:
    @pytest.mark.parametrize("model_name", MODELS)
    def test_init(self, model_name: str) -> None:
        reader = FusionELReader(model_name_or_path=model_name)
        assert isinstance(reader, FusionELReader)
        assert reader.model_name_or_path == model_name

    @pytest.mark.parametrize("model_name", MODELS)
    def test_encode(self, model_name: str) -> None:
        reader = FusionELReader(model_name_or_path=model_name)
        input_ids, attention_mask, _ = mock_input()
        fusioned_last_hidden_state, fusioned_attention_mask = reader.encode(input_ids=input_ids, attention_mask=attention_mask)
        assert fusioned_last_hidden_state.size(0) == input_ids.size(0)
        assert fusioned_last_hidden_state.size(1) == input_ids.size(1) * input_ids.size(2)
        assert fusioned_attention_mask.size() == (input_ids.size(0), input_ids.size(1) * input_ids.size(2))

    @pytest.mark.parametrize("model_name", MODELS)
    def test_forward(self, model_name: str) -> None:
        reader = FusionELReader(model_name_or_path=model_name)
        input_ids, attention_mask, labels = mock_input()
        outputs = reader(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        assert hasattr(outputs, "loss")
        assert hasattr(outputs, "logits")

    @pytest.mark.parametrize("model_name", MODELS)
    def test_generate(self, model_name: str) -> None:
        reader = FusionELReader(model_name_or_path=model_name)
        input_ids, attention_mask, _ = mock_input()
        generated_ids = reader.generate(input_ids=input_ids, attention_mask=attention_mask, max_generation_length=10)
        assert generated_ids.size(0) == input_ids.size(0)
        assert generated_ids.size(1) <= 10

    def test_save_and_load_pretrained(self) -> None:
        import tempfile
        model_name = MODELS[0]
        reader = FusionELReader(model_name_or_path=model_name)
        with tempfile.TemporaryDirectory() as tmpdir:
            reader.save_pretrained(tmpdir)
            new_reader = FusionELReader(model_name_or_path=model_name)
            new_reader.from_pretrained(tmpdir)
            assert isinstance(new_reader, FusionELReader)

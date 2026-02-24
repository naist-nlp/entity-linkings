
import pytest
import torch
from transformers import AutoModelForSeq2SeqLM

from .reader import CheckpointWrapper, EncoderWrapper, FusionDecoder

MODELS = ["google/flan-t5-small"]


def mock_input(
        batch_size: int = 2,
        num_candidates: int = 3,
        seq_length: int = 10,
        vocab_size: int = 100
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = torch.randint(0, vocab_size, (batch_size, num_candidates, seq_length))
    attention_mask = torch.ones((batch_size, num_candidates, seq_length), dtype=torch.long)
    labels = torch.randint(0, vocab_size, (batch_size, seq_length))
    return input_ids, attention_mask, labels


class TestEncoderWrapper:
    @pytest.mark.parametrize("model_name", MODELS)
    @pytest.mark.parametrize("use_checkpoint", [True, False])
    def test__init__(self, model_name: str, use_checkpoint: bool) -> None:
        t5 = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        encoder_wrapper = EncoderWrapper(t5.get_encoder(), use_checkpoint=use_checkpoint)
        assert isinstance(encoder_wrapper, EncoderWrapper)
        assert encoder_wrapper.main_input_name == "input_ids"
        assert hasattr(encoder_wrapper, "encoder")
        for module in encoder_wrapper.encoder.block:
            assert isinstance(module, CheckpointWrapper)
            assert hasattr(module, "forward")
            assert module.use_checkpoint == use_checkpoint

    @pytest.mark.parametrize("model_name", MODELS)
    @pytest.mark.parametrize("use_checkpoint", [True, False])
    def test_forward(self, model_name: str, use_checkpoint: bool) -> None:
        t5 = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        encoder_wrapper = EncoderWrapper(t5.get_encoder(), use_checkpoint=use_checkpoint)
        input_ids, attention_mask, _ = mock_input()
        encoder_wrapper.n_passages = input_ids.size(1)
        input_ids = input_ids.view(input_ids.size(0), -1)
        attention_mask = attention_mask.view(attention_mask.size(0), -1)
        outputs = encoder_wrapper(input_ids=input_ids, attention_mask=attention_mask)
        assert hasattr(outputs, "last_hidden_state")


class TestFusionDecoder:
    @pytest.mark.parametrize("model_name", MODELS)
    def test__init__(self, model_name: str) -> None:
        t5 = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model = FusionDecoder(t5.config)
        model.load_t5(t5.state_dict())
        assert isinstance(model, FusionDecoder)
        assert isinstance(model.encoder, EncoderWrapper)

    @pytest.mark.parametrize("model_name", MODELS)
    def test_forward(self, model_name: str) -> None:
        t5 = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model = FusionDecoder(t5.config)
        model.load_t5(t5.state_dict())

        input_ids, attention_mask, labels = mock_input()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        assert hasattr(outputs, "loss")
        assert hasattr(outputs, "logits")

    @pytest.mark.parametrize("model_name", MODELS)
    def test_generate(self, model_name: str) -> None:
        t5 = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model = FusionDecoder(t5.config)
        model.load_t5(t5.state_dict())
        input_ids, attention_mask, _ = mock_input()
        generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask)
        assert generated_ids.size(0) == input_ids.size(0)
        assert 3 <= generated_ids.size(1) <= 201

    @pytest.mark.parametrize("model_name", MODELS)
    def test_save_pretrained(self, model_name: str, tmp_path: pytest.TempPathFactory) -> None:
        t5 = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model = FusionDecoder(t5.config)
        model.load_t5(t5.state_dict())

        save_dir = tmp_path / "fusioned_model"
        model.save_pretrained(save_dir)
        loaded_model = FusionDecoder.from_pretrained(save_dir)
        assert isinstance(loaded_model, FusionDecoder)
        assert isinstance(loaded_model.encoder, EncoderWrapper)
        for block in loaded_model.encoder.encoder.block:
            assert isinstance(block, CheckpointWrapper)

import tempfile

import pytest
import torch
from transformers import (
    BertModel,
    DebertaV2Model,
    ModernBertModel,
    RobertaModel,
    XLMRobertaModel,
)

from .encoder import Encoder

# These cover the architectures the encoder has to work with, at a few MB each
# instead of several GB. The assertions below are about shapes and plumbing, not
# about what the weights predict. XLM-RoBERTa has no tiny build and shares its
# implementation with RoBERTa.
MODELS = [
    "hf-internal-testing/tiny-random-BertModel",
    "hf-internal-testing/tiny-random-RobertaModel",
    "hf-internal-testing/tiny-random-DebertaV2Model",
    "hf-internal-testing/tiny-random-ModernBertModel",
]

def mock_input(
        batch_size: int = 2,
        num_candidates: int = 3,
        seq_length: int = 10,
        vocab_size: int = 100
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    import torch
    input_ids = torch.randint(0, vocab_size, (batch_size, num_candidates, seq_length))
    attention_mask = torch.ones((batch_size, num_candidates, seq_length), dtype=torch.long)
    labels = torch.randint(0, num_candidates, (batch_size, ), dtype=torch.long)
    return input_ids, attention_mask, labels

class TestEncoder:
    @pytest.mark.parametrize("model_name", MODELS)
    def test__init__(self, model_name: str) -> None:
        model = Encoder(model_name)
        assert isinstance(model, Encoder)
        assert hasattr(model, "encoder") and hasattr(model, "projection") and hasattr(model, "pooler")
        assert isinstance(model.encoder, (BertModel, RobertaModel, DebertaV2Model, XLMRobertaModel, ModernBertModel))

    def test_resize_token_embeddings(self) -> None:
        model = Encoder(MODELS[0])
        old_embeddings = model.encoder.get_input_embeddings()
        old_num_tokens, embedding_dim = old_embeddings.weight.size()
        new_num_tokens = old_num_tokens + 10
        model.resize_token_embeddings(new_num_tokens)
        new_embeddings = model.encoder.get_input_embeddings()
        assert new_embeddings.weight.size() == (new_num_tokens, embedding_dim)

    @pytest.mark.parametrize("model_name", MODELS)
    def test_encode(self, model_name: str) -> None:
        model = Encoder(model_name)
        input_ids, attention_mask, _ = mock_input()
        bs, cs, length = input_ids.size()
        input_ids = input_ids.view(bs * cs, length)
        attention_mask = attention_mask.view(bs * cs, length)

        with torch.no_grad():
            pooled_output = model.encode(input_ids=input_ids, attention_mask=attention_mask)
        assert pooled_output.size(0) == input_ids.size(0)
        assert pooled_output.size(1) == model.encoder.config.hidden_size

    @pytest.mark.parametrize("model_name", MODELS)
    def test_forward(self, model_name: str) -> None:
        model = Encoder(model_name)
        input_ids, attention_mask, labels = mock_input()
        with torch.no_grad():
            loss, scores = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        assert isinstance(loss, torch.Tensor)
        assert scores.size(0) == input_ids.size(0)
        assert scores.size(1) == input_ids.size(1)

    @pytest.mark.parametrize("model_name", [MODELS[0]])
    def test_save_and_load(self, model_name: str) -> None:
        model = Encoder(model_name)
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            loaded_model = Encoder.from_pretrained(tmpdir)
            assert isinstance(loaded_model, Encoder)

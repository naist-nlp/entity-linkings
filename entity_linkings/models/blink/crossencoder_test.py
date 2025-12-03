import pytest
import torch
from transformers import (
    BertModel,
    DebertaV2Model,
    ModernBertModel,
    RobertaModel,
    XLMRobertaModel,
)

from .crossencoder import CrossEncoder

MODELS = [
    "google-bert/bert-base-uncased",
    "FacebookAI/xlm-roberta-base",
    "microsoft/deberta-v3-base",
    "FacebookAI/roberta-base",
    "answerdotai/ModernBERT-base",
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
    labels = torch.randint(0, 2, (batch_size, num_candidates))
    return input_ids, attention_mask, labels

class TestCrossEncoder:
    @pytest.mark.parametrize("model_name", MODELS)
    def test__init__(self, model_name: str) -> None:
        model = CrossEncoder(model_name)
        assert isinstance(model, CrossEncoder)
        assert hasattr(model, "encoder") and hasattr(model, "projection") and hasattr(model, "pooler")
        assert isinstance(model.encoder, (BertModel, RobertaModel, DebertaV2Model, XLMRobertaModel, ModernBertModel))

    def test_resize_token_embeddings(self) -> None:
        model = CrossEncoder(MODELS[0])
        old_embeddings = model.encoder.get_input_embeddings()
        old_num_tokens, embedding_dim = old_embeddings.weight.size()
        new_num_tokens = old_num_tokens + 10
        model.resize_token_embeddings(new_num_tokens)
        new_embeddings = model.encoder.get_input_embeddings()
        assert new_embeddings.weight.size() == (new_num_tokens, embedding_dim)

    @pytest.mark.parametrize("model_name", MODELS)
    def test_encode(self, model_name: str) -> None:
        model = CrossEncoder(model_name)
        input_ids, attention_mask, _ = mock_input()
        bs, cs, length = input_ids.size()
        input_ids = input_ids.view(bs * cs, length)
        attention_mask = attention_mask.view(bs * cs, length)

        with torch.no_grad():
            pooled_output = model.encode(input_ids=input_ids, attention_mask=attention_mask)
        assert pooled_output.size(0) == input_ids.size(0)
        assert pooled_output.size(1) == model.encoder.config.hidden_size

    def test_score(self) -> None:
        model = CrossEncoder(MODELS[0])
        input_ids, attention_mask, _ = mock_input()
        bs, cs, length = input_ids.size()
        input_ids = input_ids.view(bs * cs, length)
        attention_mask = attention_mask.view(bs * cs, length)

        with torch.no_grad():
            scores = model.score(input_ids=input_ids, attention_mask=attention_mask)
        assert scores.size(0) == input_ids.size(0)

    @pytest.mark.parametrize("model_name", MODELS)
    def test_forward(self, model_name: str) -> None:
        model = CrossEncoder(model_name)
        input_ids, attention_mask, labels = mock_input()
        with torch.no_grad():
            loss, scores = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        assert isinstance(loss, torch.Tensor)
        assert scores.size(0) == input_ids.size(0)
        assert scores.size(1) == input_ids.size(1)

    @pytest.mark.parametrize("model_name", [MODELS[0]])
    def test_save_and_load(self, model_name: str) -> None:
        model = CrossEncoder(model_name)
        model.save_pretrained("./test_crossencoder_save")
        loaded_model = CrossEncoder.from_pretrained("./test_crossencoder_save")
        assert isinstance(loaded_model, CrossEncoder)

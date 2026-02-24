import tempfile

import pytest
import torch

from .encoder import TextEmbeddingModel

TEXT_EMBEDDING_MODELS = [
    # TODO: add more models
    "intfloat/e5-small",
    "intfloat/multilingual-e5-small"
]


def mock_model_inputs(
        batch_size: int = 2,
        num_candidates: int = 4,
        seq_length: int = 10,
        vocab_size: int = 100,
    ) -> dict[str, torch.Tensor]:
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
    candidates_input_ids = torch.randint(0, vocab_size, (num_candidates, seq_length))
    candidates_attention_mask = torch.ones((num_candidates, seq_length), dtype=torch.long)
    labels = torch.randint(0, num_candidates, (batch_size,))
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "candidates_input_ids": candidates_input_ids,
        "candidates_attention_mask": candidates_attention_mask,
        "labels": labels
    }


class TestTextEmbeddingModel:
    def test__init__(self) -> None:
        model = TextEmbeddingModel(TEXT_EMBEDDING_MODELS[0])
        assert isinstance(model, TextEmbeddingModel)
        assert model.distance == "inner_product"
        assert model.config == {
            "model_name_or_path": TEXT_EMBEDDING_MODELS[0],
            "pooling": "mean",
            "distance": "inner_product",
            "temperature": 1.0
        }

    @pytest.mark.parametrize("model_name", TEXT_EMBEDDING_MODELS)
    def test_encode(self, model_name: str) -> None:
        model = TextEmbeddingModel(model_name)
        model_inputs = mock_model_inputs()
        with torch.no_grad():
            mention_outputs = model.encode(
                model_inputs["input_ids"],
                model_inputs["attention_mask"],
            )
            assert mention_outputs.size() == (2, 384)

            candidate_outputs = model.encode(
                model_inputs["candidates_input_ids"],
                model_inputs["candidates_attention_mask"],
            )
            assert candidate_outputs.size() == (4, 384)

    @pytest.mark.parametrize("model_name", TEXT_EMBEDDING_MODELS)
    def test_forward(self, model_name: str) -> None:
        model = TextEmbeddingModel(model_name, pooling="mean", distance="euclidean")
        model_inputs = mock_model_inputs()
        with torch.no_grad():
            loss, logits = model(**model_inputs)
            assert loss.size() == torch.Size([])
            assert logits.size() == (2, 4)

    def test_resize_token_embeddings(self) -> None:
        model = TextEmbeddingModel(TEXT_EMBEDDING_MODELS[0], pooling="mean", distance="cosine")
        model.resize_token_embeddings(100)
        assert model.encoder.config.vocab_size == 100

    def test_save_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = TextEmbeddingModel(TEXT_EMBEDDING_MODELS[0], pooling="mean", distance="cosine")
            model.save_pretrained(tmpdir)
            new_model = TextEmbeddingModel.from_pretrained(tmpdir)
            assert new_model.config == model.config

import tempfile

import pytest
import torch

from .fevry import SpanClassifier

# These cover the architectures the classifier has to work with, at a few MB each
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
        seq_length: int = 10,
        num_candidates: int = 3,
        num_entities: int = 10,
        vocab_size: int = 100
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
    start_positions = torch.randint(0, seq_length // 2, (batch_size,))
    end_positions = start_positions + torch.randint(1, seq_length // 2, (batch_size,))
    labels = torch.randint(0, num_candidates, (batch_size,))
    candidates = torch.randint(0, num_entities, (batch_size, num_candidates))
    return input_ids, attention_mask, start_positions, end_positions, labels, candidates


class TestSpanClassifier:
    @pytest.mark.parametrize("model_name", MODELS)
    def test_init_(self, model_name: str) -> None:
        encoder = SpanClassifier(model_name_or_path=model_name, num_entities=10)
        assert isinstance(encoder, SpanClassifier)
        # The projection takes the start and end states side by side.
        assert encoder.projection.in_features == encoder.encoder.config.hidden_size * 2
        assert encoder.projection.out_features == 256
        assert encoder.entity_embeddings.num_embeddings == 10
        assert encoder.entity_embeddings.embedding_dim == 256


    @pytest.mark.parametrize("model_name", MODELS)
    def test_encode(self, model_name: str) -> None:
        encoder = SpanClassifier(model_name_or_path=model_name, num_entities=10)
        input_ids, attention_mask, start_positions, end_positions, _, _ = mock_input()
        with torch.no_grad():
            span_embeds = encoder.encode(
                input_ids=input_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions,
            )
            assert span_embeds.size() == (input_ids.size(0), encoder.encoder.config.hidden_size * 2)

    @pytest.mark.parametrize("model_name", MODELS)
    def test_forward(self, model_name: str) -> None:
        encoder = SpanClassifier(model_name_or_path=model_name, num_entities=10)
        input_ids, attention_mask, start_positions, end_positions, labels, candidates = mock_input()
        with torch.no_grad():
            _, logits = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions,
                labels=labels,
                candidates_ids=candidates
            )
            assert logits.size() == (2, 3)

    def test_save_and_load_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            encoder = SpanClassifier(model_name_or_path=MODELS[0], num_entities=20)
            encoder.save_pretrained(tmpdir)
            new_encoder = SpanClassifier.from_pretrained(tmpdir)
            assert new_encoder.projection.in_features == encoder.encoder.config.hidden_size * 2
            assert new_encoder.projection.out_features == 256
            assert new_encoder.entity_embeddings.num_embeddings == 20
            assert new_encoder.entity_embeddings.embedding_dim == 256

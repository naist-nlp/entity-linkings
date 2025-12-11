
import tempfile

import pytest
import torch

from .qa_span_classifier import QASpanClassifier

MODELS = [
    "google-bert/bert-base-uncased",
    "allenai/longformer-base-4096",
]

def mock_input(
        batch_size: int = 5,
        seq_length: int = 10,
        num_candidates: int = 3,
        vocab_size: int = 100
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
    start_positions = torch.randint(0, seq_length // 2, (batch_size, num_candidates))
    end_positions = start_positions + torch.randint(1, seq_length // 2, (batch_size, num_candidates))
    candidate_offsets = torch.stack([start_positions, end_positions], dim=-1) # (batch_size, num_candidates, 2)
    labels = torch.stack([start_positions[:, 0], end_positions[:, 0]], dim=0)  # (2, batch_size, )
    return input_ids, attention_mask, start_positions, end_positions, candidate_offsets, labels


class TestQASpanClassifier:
    def test_init_(self) -> None:
        encoder = QASpanClassifier(model_name_or_path=MODELS[0])
        assert isinstance(encoder, QASpanClassifier)

    @pytest.mark.parametrize("model_name", MODELS)
    @pytest.mark.parametrize("modify_global_attention", [1, 2])
    def test_forward(self, model_name: str, modify_global_attention: int) -> None:
        model = QASpanClassifier(
            model_name_or_path=model_name,
            modify_global_attention=modify_global_attention
        )
        input_ids, attention_mask, _, _, candidate_offsets, labels = mock_input()
        with torch.no_grad():
            _, logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                candidates_offsets=candidate_offsets,
                labels=labels,
            )
            assert logits.size() == (2, 5, 10)

    def test_save_and_load_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            encoder = QASpanClassifier(model_name_or_path=MODELS[0])
            encoder.resize_token_embeddings(150)
            encoder.save_pretrained(tmpdir)
            new_encoder = QASpanClassifier.from_pretrained(tmpdir)
            assert new_encoder.model.get_input_embeddings().weight.size(0) == 150

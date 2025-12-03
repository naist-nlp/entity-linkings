import pytest
import torch
from transformers import AutoTokenizer

from .utils import (
    concat_pooler,
    first_token_pooler,
    get_pooler,
    last_token_pooler,
    load_model,
    log_marginal_likelihood,
    mean_tokens_pooler,
)

MODELS = [
    "google-bert/bert-base-uncased",
    "FacebookAI/xlm-roberta-base",
    "microsoft/deberta-v3-base",
    "FacebookAI/roberta-base",
    "answerdotai/ModernBERT-base",
    "google-t5/t5-small",
    "intfloat/multilingual-e5-small"
]

def test_log_marginal_likelihood() -> None:
    logits = torch.tensor([[0.2, 0.8, 1.0], [1.0, 2.0, 3.0]])
    labels = torch.tensor([[0, 1, 0], [0, 0, 1]])
    loss = log_marginal_likelihood(logits, labels)
    loss = -((torch.logsumexp(torch.tensor([[-10000.0, 0.8, -10000.0]]), -1) - torch.logsumexp(torch.tensor([[0.2, 0.8, 1.0]]), -1)) +
             (torch.logsumexp(torch.tensor([[-10000.0, -10000.0, 3.0]]), -1) - torch.logsumexp(torch.tensor([[1.0, 2.0, 3.0]]), -1))).item()
    computed_loss = log_marginal_likelihood(logits, labels).item()
    assert abs(loss - computed_loss) < 1e-6


@pytest.mark.parametrize("model_name", MODELS)
def test_load_model(model_name: str) -> None:
    model = load_model(model_name)
    assert hasattr(model, "config")
    assert hasattr(model.config, "hidden_size")
    assert hasattr(model, "forward")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer("This is a test", return_tensors="pt")
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    assert outputs.last_hidden_state.shape[0] == 1
    assert outputs.last_hidden_state.shape[1] == inputs["input_ids"].shape[1]
    assert outputs.last_hidden_state.shape[2] == model.config.hidden_size

    model.save_pretrained("test_model")
    new_model = load_model("test_model")
    new_model.eval()
    with torch.no_grad():
        new_outputs = new_model(**inputs)
    assert torch.allclose(outputs.last_hidden_state, new_outputs.last_hidden_state, atol=1)


@pytest.mark.parametrize("pooling", ["first", "mean", "last"])
def test_get_pooler(pooling: str | None) -> None:
    pooler = get_pooler(pooling)
    if pooling == "first":
        assert pooler is first_token_pooler
    elif pooling == "mean":
        assert pooler is mean_tokens_pooler
    elif pooling == "last":
        assert pooler is last_token_pooler


def test_first_token_pooler() -> None:
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    tokenizer.add_tokens(["[START_ENT]", "[END_ENT]"])
    attention_mask = tokenizer(["Steve Jobs was found [START_ENT] Apple [END_ENT]."], return_tensors="pt", padding='max_length').attention_mask
    assert attention_mask.size() == (1, 512)
    embeddings = torch.randn(1, 512, 200)
    pooled_output = first_token_pooler(embeddings, attention_mask)
    assert pooled_output.size() == (1, 200)


def test_mean_token_pooler() -> None:
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    tokenizer.add_tokens(["[START_ENT]", "[END_ENT]"])
    attention_mask = tokenizer(["Steve Jobs was found [START_ENT] Apple [END_ENT]."], return_tensors="pt", padding='max_length').attention_mask
    assert attention_mask.size() == (1, 512)
    embeddings = torch.randn(1, 512, 200)
    pooled_output = mean_tokens_pooler(embeddings, attention_mask)
    assert pooled_output.size() == (1, 200)


def test_last_token_pooler() -> None:
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    tokenizer.add_tokens(["[START_ENT]", "[END_ENT]"])
    attention_mask = tokenizer(["Steve Jobs was found [START_ENT] Apple [END_ENT]."], return_tensors="pt", padding='max_length').attention_mask
    assert attention_mask.size() == (1, 512)
    embeddings = torch.randn(1, 512, 200)
    pooled_output = last_token_pooler(embeddings, attention_mask)
    assert pooled_output.size() == (1, 200)


def test_concat_token_pooler() -> None:
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    tokenizer.add_tokens(["[START_ENT]", "[END_ENT]"])
    attention_mask = tokenizer(["Steve Jobs was found [START_ENT] Apple [END_ENT]."], return_tensors="pt", padding='max_length').attention_mask
    assert attention_mask.size() == (1, 512)
    embeddings = torch.randn(1, 512, 200)
    pooled_output = concat_pooler(embeddings, attention_mask)
    assert pooled_output.size() == (1, 400)

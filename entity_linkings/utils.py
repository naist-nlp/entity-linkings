from dataclasses import dataclass
from typing import Any, Callable

import torch
import yaml
from transformers import AutoConfig, AutoModel, PreTrainedModel, T5EncoderModel

ENCODER_DECODER_MODELS = {
    "T5Config": T5EncoderModel,
}


@dataclass
class BaseSystemOutput:
    query: str
    start: int
    end: int
    id: str


def read_yaml(path: str) -> dict[str, Any]:
    with open(path, 'r') as yml:
        config = yaml.safe_load(yml)
    return config


def load_model(model_name_or_path: str) -> PreTrainedModel:
    model_config = AutoConfig.from_pretrained(model_name_or_path)

    if type(model_config).__name__ in ENCODER_DECODER_MODELS:
        model_cls = ENCODER_DECODER_MODELS.get(type(model_config).__name__, None)
        if not model_cls:
            raise NotImplementedError(f"Encoder-decoder models are not supported. {model_config}")
        model = model_cls.from_pretrained(model_name_or_path, config=model_config)
    else:
        if model_config.is_encoder_decoder:
            raise NotImplementedError(f"Encoder-decoder models are not supported. {model_config}")
        model = AutoModel.from_pretrained(model_name_or_path, config=model_config)
    return model


def get_pooler(pooling: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Get the pooling function based on the specified method.
    Args:
        - pooling (`str`):
            The pooling method to use. Options are 'first', 'mean', or 'last'.
    Returns:
        - `Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`: The pooling function.
    """
    if pooling == 'first':
        return first_token_pooler
    elif pooling == 'mean':
        return mean_tokens_pooler
    elif pooling == 'last':
        return last_token_pooler
    elif pooling == 'concat':
        return concat_pooler
    else:
        raise ValueError(f"Invalid pooling method: {pooling}. Choose either 'first', 'mean', or 'last'.")


def first_token_pooler(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Pooling using the first token ([CLS] or [BOS]) representation.
    Args:
        - last_hidden_states (`torch.Tensor`):
            The last hidden states from the model of shape (batch_size, sequence_length, hidden_size).
        - attention_mask (`torch.Tensor`):
            The attention mask of shape (batch_size, sequence_length), where 1 indicates valid tokens
    Returns:
        - `torch.Tensor`: The pooled output of shape (batch_size, hidden_size).
    """
    if attention_mask[:, 0].sum() == 0 and attention_mask[:, -1].sum() > 0:
        # left padding is used, use the first non-padding token as the last token.
        seq_lens = attention_mask.sum(dim=1)
        return last_hidden_states[:, -seq_lens, :]
    else:
        # right padding is used. use the first token as the start token.
        return last_hidden_states[:, 0, :]


def mean_tokens_pooler(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Pooling by averaging the token representations, weighted by the attention mask.
    Args:
        - last_hidden_states (`torch.Tensor`):
            The last hidden states from the model of shape (batch_size, sequence_length, hidden_size).
        - attention_mask (`torch.Tensor`):
            The attention mask of shape (batch_size, sequence_length), where 1 indicates valid tokens and 0 indicates padding.
    Returns:
        - `torch.Tensor`: The pooled output of shape (batch_size, hidden_size).
    """
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def last_token_pooler(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Pooling using the last token representation.
    Args:
        - last_hidden_states (`torch.Tensor`):
            The last hidden states from the model of shape (batch_size, sequence_length, hidden_size).
        - attention_mask (`torch.Tensor`):
            The attention mask of shape (batch_size, sequence_length), where 1 indicates valid tokens and 0 indicates padding.
    Returns:
        - `torch.Tensor`: The pooled output of shape (batch_size, hidden_size).
    """
    if attention_mask[:, 0].sum() == 0 and attention_mask[:, -1].sum() > 0:
        # left padding is used, use the last token as the last token.
        return last_hidden_states[:, -1, :]
    else:
        # right padding is used. use the last non-padding token as the last token.
        seq_lens = attention_mask.sum(dim=1) - 1
        return last_hidden_states[torch.arange(last_hidden_states.size(0)), seq_lens, :]


def concat_pooler(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Pooling by concatenating the first and last token representations.
    Args:
        - last_hidden_states (`torch.Tensor`):
            The last hidden states from the model of shape (batch_size, sequence_length, hidden_size).
    Returns:
        - `torch.Tensor`: The pooled output of shape (batch_size, 2 * hidden_size).
    """
    first_token = first_token_pooler(last_hidden_states, attention_mask)
    last_token = last_token_pooler(last_hidden_states, attention_mask)
    return torch.cat((first_token, last_token), dim=1)


def calculate_top1_accuracy(num_corrects: int, num_golds: int) -> dict[str, float]:
    return {
        "top1_accuracy": num_corrects / num_golds if num_golds > 0 else 0.
    }


def calculate_recall_mrr(predictions: list[dict[str, Any]]) -> dict[str, int | float]:
    true, tp_1, tp_10, tp_50, tp_100, reciprocal_rank = 0, 0, 0, 0, 0, 0.
    for prediction in predictions:
        true += 1
        best_rank = 0
        indices = [result['id'] for result in prediction['predict']]
        for label in prediction['gold']:
            if label in indices:
                rank = indices.index(label) + 1
                if rank < best_rank or best_rank == 0:
                    best_rank = rank
        if best_rank > 0:
            if best_rank == 1:
                tp_1 += 1
            if best_rank <= 10:
                tp_10 += 1
            if best_rank <= 50:
                tp_50 += 1
            if best_rank <= 100:
                tp_100 += 1
            reciprocal_rank += 1 / best_rank

    return {
        "recall@1": tp_1 / true if true > 0 else 0.,
        "recall@10": tp_10 / true if true > 0 else 0.,
        "recall@50": tp_50 / true if true > 0 else 0.,
        "recall@100": tp_100 / true if true > 0 else 0.,
        "mrr@100": reciprocal_rank / true if true > 0. else 0.
    }


def calculate_inkb_f1(predictions: list[list[dict[str, Any]]], golds: list[list[dict[str, Any]]]) -> dict[str, int | float]:
    tp, t, p = 0, 0, 0
    assert len(predictions) == len(golds)
    for prediction, gold in zip(predictions, golds):
        gold_ids = {(gold['start'], gold['end']): gold['label'] for gold in gold}
        pred_ids = {(result['start'], result['end']): result['label'] for result in prediction}
        for p_span, p_label in pred_ids.items():
            if p_span in gold_ids and len(set(p_label) & set(gold_ids[p_span])) > 0:
                tp += 1
        t += len(gold_ids)
        p += len(pred_ids)

    precision = tp / p if p > 0 else 0.
    recall = tp / t if t > 0 else 0.
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

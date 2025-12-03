from typing import Callable

import torch
from transformers import AutoConfig, AutoModel, PreTrainedModel, T5EncoderModel

ENCODER_DECODER_MODELS = {
    "T5Config": T5EncoderModel,
}


def log_marginal_likelihood(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    gold_scores = logits.masked_fill(~(labels.bool()), -10000)
    gold_log_sum_exp = torch.logsumexp(gold_scores, -1)
    all_log_sum_exp = torch.logsumexp(logits, -1)
    gold_log_probs = gold_log_sum_exp - all_log_sum_exp
    loss = -gold_log_probs.sum()
    return loss


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


import json
import os
from typing import Optional

import torch
import torch.nn as nn

from entity_linkings.models.utils import get_pooler, load_model, log_marginal_likelihood


class CrossEncoder(nn.Module):
    '''
    Cross Encoder for entity disambiguation.
    '''

    # to suppress an AttributeError when training
    _keys_to_ignore_on_save = None
    def __init__(self, model_name_or_path: str, pooling: str = 'first') -> None:
        super().__init__()
        self.encoder = load_model(model_name_or_path)
        self.projection = nn.Linear(self.encoder.config.hidden_size, 1)
        self.pooler = get_pooler(pooling)
        self.config = {"model_name_or_path": model_name_or_path, "pooling": pooling}

    def resize_token_embeddings(self, new_num_tokens: int) -> None:
        self.encoder.resize_token_embeddings(new_num_tokens)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            model_inputs["token_type_ids"] = token_type_ids
        encoder_outputs = self.encoder(**model_inputs)
        last_hidden_state = encoder_outputs.last_hidden_state
        pooled_output = self.pooler(last_hidden_state, attention_mask)
        return pooled_output # (bs, hidden_size)

    def score(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.encode(input_ids, attention_mask, token_type_ids)  # (bs, hidden_size)
        scores = self.projection(outputs).squeeze(-1)  # (bs, )
        return scores

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None) -> tuple[torch.Tensor|None, torch.Tensor]:
        bs, cs, length = input_ids.size()
        input_ids = input_ids.view(bs * cs, length)
        attention_mask = attention_mask.view(bs * cs, length)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(bs * cs, length)

        scores = self.score(input_ids, attention_mask, token_type_ids) # (bs * cs, )
        scores = scores.view(bs, cs) # (bs, cs)

        if labels is None:
            return (None, scores)

        #  log marginal likelihood
        loss = log_marginal_likelihood(scores, labels)
        return (loss, scores)

    def save_pretrained(self, save_directory: str) -> None:
        self.encoder.save_pretrained(save_directory)
        torch.save(self.projection.state_dict(), os.path.join(save_directory, "projection.pt"))
        model_config = {"model_name_or_path": self.config["model_name_or_path"], "pooling": self.config["pooling"]}
        json.dump(model_config, open(os.path.join(save_directory, "model_config.json"), "w"), indent=2, ensure_ascii=False)

    @classmethod
    def from_pretrained(cls, load_directory: str) -> "CrossEncoder":
        model_config = json.load(open(os.path.join(load_directory, "model_config.json")))
        model_config["model_name_or_path"] = load_directory
        model = cls(**model_config)
        model.projection.load_state_dict(torch.load(os.path.join(load_directory, "projection.pt")))
        return model

import json
import os
from typing import Optional

import torch
import torch.nn as nn

from entity_linkings.utils import load_model


class SpanClassifier(nn.Module):

    # to suppress an AttributeError when training
    _keys_to_ignore_on_save = None
    def __init__(self, model_name_or_path: str, num_entities: int, projection_dim: int = 256) -> None:
        super().__init__()
        self.encoder = load_model(model_name_or_path)

        self.projection = torch.nn.Linear(self.encoder.config.hidden_size * 2, projection_dim)
        classifier_dropout = (
            self.encoder.config.classifier_dropout
            if hasattr(self.encoder.config, "classifier_dropout") and self.encoder.config.classifier_dropout is not None
            else self.encoder.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        self.entity_embeddings = nn.Embedding(num_entities, projection_dim)
        self.entity_bias = nn.Embedding(num_entities, 1)
        self.config = {"model_name_or_path": model_name_or_path, "num_entities": num_entities, "projection_dim": projection_dim}

    def resize_token_embeddings(self, new_num_tokens: int) -> None:
        self.encoder.resize_token_embeddings(new_num_tokens)

    def encode(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            start_positions: torch.Tensor,
            end_positions: torch.Tensor,
            token_type_ids: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
        bs, _ = input_ids.size()
        if token_type_ids is not None:
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = encoder_outputs.last_hidden_state # (bs, length, hidden_size)

        span_start_embeds = last_hidden_state[torch.arange(bs), start_positions, :]
        span_end_embeds = last_hidden_state[torch.arange(bs), end_positions, :]
        span_embeds = torch.cat([span_start_embeds, span_end_embeds], dim=-1)

        return span_embeds # (bs, hidden_size * 2)

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: Optional[torch.Tensor] = None,
            start_positions: Optional[torch.Tensor] = None,
            end_positions: Optional[torch.Tensor] = None,
            candidates_ids: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
        ) -> tuple[Optional[torch.Tensor], torch.Tensor]:

        span_embeds = self.encode(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions,
            token_type_ids=token_type_ids
        )  # (bs, hidden_size * 2)

        span_embeds = self.projection(span_embeds) # (bs, 256)
        span_embeds = self.dropout(span_embeds) # (bs, 256)

        candidate_embeddings = self.entity_embeddings(candidates_ids)  # (bs, num_candidates, 256)
        candidate_bias = self.entity_bias(candidates_ids).squeeze(-1)  # (bs, num_candidates

        logits = torch.bmm(
            span_embeds.unsqueeze(1),
            candidate_embeddings.transpose(1, 2),
        ).squeeze(1) + candidate_bias  # (bs, num_candidates)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)

        return (loss, logits)

    def save_pretrained(self, save_directory: str) -> None:
        self.encoder.save_pretrained(save_directory)
        torch.save(self.projection.state_dict(), os.path.join(save_directory, "projection.pt"))
        torch.save(self.entity_embeddings.state_dict(), os.path.join(save_directory, 'embedding.pt'))
        torch.save(self.entity_bias.state_dict(), os.path.join(save_directory, 'bias.pt'))
        model_config = {
            "model_name_or_path": self.config["model_name_or_path"],
            "num_entities": self.config["num_entities"],
        }
        json.dump(model_config, open(os.path.join(save_directory, "model_config.json"), "w"), indent=2)

    @classmethod
    def from_pretrained(cls, load_directory: str) -> "SpanClassifier":
        model_config = json.load(open(os.path.join(load_directory, "model_config.json")))
        model_config["model_name_or_path"] = load_directory
        model = cls(**model_config)
        model.projection.load_state_dict(torch.load(os.path.join(load_directory, "projection.pt")))
        model.entity_embeddings.load_state_dict(torch.load(os.path.join(load_directory, "embedding.pt")))
        model.entity_bias.load_state_dict(torch.load(os.path.join(load_directory, 'bias.pt')))
        return model

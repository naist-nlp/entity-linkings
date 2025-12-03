import json
import os
from typing import Optional

import torch
import torch.nn as nn

from entity_linkings.models.utils import load_model


class SpanClassifier(nn.Module):

    # to suppress an AttributeError when training
    _keys_to_ignore_on_save = None
    def __init__(self, model_name_or_path: str, num_entities: int) -> None:
        super().__init__()
        self.encoder = load_model(model_name_or_path)
        self.classifier = torch.nn.Linear(self.encoder.config.hidden_size * 2, num_entities)
        self.config = {"model_name_or_path": model_name_or_path, "num_entities": num_entities}

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
            input_ids,attention_mask, start_positions, end_positions, token_type_ids
        )  # (bs, hidden_size * 2)

        scores = self.classifier(span_embeds)  # (bs, num_entities)

        masked_scores = torch.ones(scores.size(), requires_grad=True, device='cuda' if torch.cuda.is_available() else 'cpu')
        masked_scores = masked_scores * -torch.inf
        if candidates_ids is not None:
            for idx in range(candidates_ids.size(0)):
                masked_scores[idx, candidates_ids[idx]] = scores[idx, candidates_ids[idx]]
        else:
            masked_scores = scores # (bs, num_entities)

        if labels is None:
            return (None, scores)

        loss = nn.functional.cross_entropy(masked_scores, labels)
        return (loss, scores)

    def save_pretrained(self, save_directory: str) -> None:
        self.encoder.save_pretrained(save_directory)
        torch.save(self.classifier.state_dict(), os.path.join(save_directory, "classifier.pt"))
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
        model.classifier.load_state_dict(torch.load(os.path.join(load_directory, "classifier.pt")))
        return model

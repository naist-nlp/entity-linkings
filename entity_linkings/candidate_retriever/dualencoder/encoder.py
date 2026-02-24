import json
from typing import Optional

import torch
import torch.nn as nn

from entity_linkings.utils import get_pooler, load_model


class DualBERTModel(nn.Module):
    # to suppress an AttributeError when training
    _keys_to_ignore_on_save = None

    def __init__(self, model_name_or_path: str, pooling: str = "first", distance: str = "inner_product", temperature: float = 1.0) -> None:
        """Dual-Encoder BERT Model for Span Retrieval
        Args:
            model_name_or_path (str): Pre-trained model name or path
            pooling (str): Pooling method for obtaining fixed-size representations ('first', 'mean', "last")
            distance (str): Similarity metric for retrieval ('inner_product', 'cosine', 'euclidean')
            temperature (float): Temperature parameter for scaling similarity scores
        """
        super().__init__()
        self.mention_encoder = load_model(model_name_or_path)
        self.candidate_encoder = load_model(model_name_or_path)
        self.hidden_size = self.mention_encoder.config.hidden_size * 2 if pooling == "concat" else self.mention_encoder.config.hidden_size
        self.pooler = get_pooler(pooling)
        if distance not in ['inner_product', 'cosine', 'euclidean']:
            raise ValueError("Distance must be 'inner_product' or 'cosine'.")
        self.distance = distance
        self.temperature = temperature
        self.config = {
            "model_name_or_path": model_name_or_path,
            "pooling": pooling,
            "distance": distance,
            "temperature": temperature
        }

    def encode_mention(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            model_inputs["token_type_ids"] = token_type_ids
        outputs = self.mention_encoder(**model_inputs).last_hidden_state
        outputs = self.pooler(outputs, attention_mask)
        return outputs

    def encode_candidate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            model_inputs["token_type_ids"] = token_type_ids
        outputs = self.candidate_encoder(**model_inputs).last_hidden_state
        outputs = self.pooler(outputs, attention_mask)
        return outputs

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: Optional[torch.Tensor] = None,
            candidates_input_ids: Optional[torch.Tensor] = None,
            candidates_attention_mask: Optional[torch.Tensor] = None,
            candidates_token_type_ids: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
        ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        mention_embeddings = self.encode_mention(input_ids, attention_mask, token_type_ids) # (bs, hidden_size)
        candidates_embeddings = self.encode_candidate(candidates_input_ids, candidates_attention_mask, candidates_token_type_ids) # (cs, hidden_size)

        if self.distance == "euclidean":
            scores = torch.cdist(mention_embeddings, candidates_embeddings, p=2)  # (bs, cs)
        else:
            if self.distance == "cosine":
                mention_embeddings = nn.functional.normalize(mention_embeddings, p=2, dim=1)
                candidates_embeddings = nn.functional.normalize(candidates_embeddings, p=2, dim=1)
            scores = mention_embeddings @ candidates_embeddings.T  # (bs, cs)

        if labels is None:
            return (None, scores)

        loss = torch.nn.functional.cross_entropy(scores, labels, reduction="mean")
        return (loss, scores)

    def resize_token_embeddings(self, new_num_tokens: int) -> None:
        self.mention_encoder.resize_token_embeddings(new_num_tokens)
        self.candidate_encoder.resize_token_embeddings(new_num_tokens)

    def save_pretrained(self, save_directory: str) -> None:
        self.mention_encoder.save_pretrained(f"{save_directory}/context_model")
        self.candidate_encoder.save_pretrained(f"{save_directory}/candidate_model")
        json.dump(self.config, open(f"{save_directory}/encoder_config.json", "w"), indent=2, ensure_ascii=False)

    @classmethod
    def from_pretrained(cls, load_directory: str) -> "DualBERTModel":
        config = json.load(open(f"{load_directory}/encoder_config.json", "r", encoding="utf-8"))
        config["model_name_or_path"] = config["model_name_or_path"]
        model = cls(**config)
        model.mention_encoder = load_model(f"{load_directory}/context_model")
        model.candidate_encoder = load_model(f"{load_directory}/candidate_model")
        return model

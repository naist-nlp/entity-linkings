import io
import json
from collections import OrderedDict
from typing import Optional

import requests
import torch
import torch.nn as nn

from entity_linkings.models.utils import get_pooler, load_model


class SpanEncoderModelBase(nn.Module):
    def __init__(self, model_name_or_path: str, pooling: str = "mean", distance: str = "inner_product", temperature: float = 1.0) -> None:
        super().__init__()

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: Optional[torch.Tensor] = None,
            candidates_input_ids: Optional[torch.Tensor] = None,
            candidates_attention_mask: Optional[torch.Tensor] = None,
            candidates_token_type_ids: Optional[torch.Tensor] = None,
            hard_negatives_input_ids: Optional[torch.Tensor] = None,
            hard_negatives_attention_mask: Optional[torch.Tensor] = None,
            hard_negatives_token_type_ids: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
        ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        raise NotImplementedError

    def resize_token_embeddings(self, new_num_tokens: int) -> None:
        raise NotImplementedError

    def save_pretrained(self, save_directory: str) -> None:
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls, load_directory: str) -> "SpanEncoderModelBase":
        raise NotImplementedError


class TextEmbeddingModel(SpanEncoderModelBase):
    def __init__(self, model_name_or_path: str, pooling: str = "mean", distance: str = "inner_product", temperature: float = 1.0) -> None:
        """Text Embedding Model for Span Retrieval
        Args:
            model_name_or_path (str): Pre-trained model name or path
            pooling (str): Pooling method for obtaining fixed-size representations ('first', 'mean', "last")
            distance (str): Similarity metric for retrieval ('inner_product', 'cosine', 'euclidean')
            temperature (float): Temperature parameter for scaling similarity scores
        """
        super().__init__(model_name_or_path, pooling, distance, temperature)
        self.encoder = load_model(model_name_or_path)
        self.hidden_size = self.encoder.config.hidden_size * 2 if pooling == "concat" else self.encoder.config.hidden_size
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

    def encode_candidate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.encode(input_ids, attention_mask, token_type_ids)

    def encode_mention(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.encode(input_ids, attention_mask, token_type_ids)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            model_inputs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**model_inputs).last_hidden_state
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
            hard_negatives_input_ids: Optional[torch.Tensor] = None,
            hard_negatives_attention_mask: Optional[torch.Tensor] = None,
            hard_negatives_token_type_ids: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
        ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        mention_embeddings = self.encode(input_ids, attention_mask, token_type_ids)
        candidates_embeddings = self.encode(candidates_input_ids, candidates_attention_mask, candidates_token_type_ids)

        bs, hs = candidates_embeddings.size(0), candidates_embeddings.size(-1)
        candidates_embeddings = candidates_embeddings.unsqueeze(0).repeat(bs, 1, 1)

        if hard_negatives_input_ids is not None and hard_negatives_attention_mask is not None:
            hard_negatives_embeddings = self.encode(hard_negatives_input_ids, hard_negatives_attention_mask, hard_negatives_token_type_ids)
            hard_negatives_embeddings = hard_negatives_embeddings.view([bs, -1, hs])
            candidates_embeddings = torch.cat([candidates_embeddings, hard_negatives_embeddings], dim=1)

        if self.distance == 'inner_product':
            scores = torch.bmm(mention_embeddings.unsqueeze(1), candidates_embeddings.transpose(1, -1)).squeeze(1)
        elif self.distance == 'cosine':
            mention_embeddings_norm = mention_embeddings.unsqueeze(1) / torch.norm(mention_embeddings.unsqueeze(1), dim=2, keepdim=True)
            candidates_embeddings_norm = candidates_embeddings / torch.norm(candidates_embeddings, dim=2, keepdim=True)
            scores = torch.bmm(mention_embeddings_norm, candidates_embeddings_norm.transpose(1, -1)).squeeze(1)
        else:
            scores = torch.cdist(mention_embeddings.unsqueeze(1), candidates_embeddings).squeeze(1)

        if labels is None:
            return (None, scores)

        loss = torch.nn.functional.cross_entropy(scores, labels, reduction="mean")
        return (loss, scores)

    def resize_token_embeddings(self, new_num_tokens: int) -> None:
        self.encoder.resize_token_embeddings(new_num_tokens)

    def save_pretrained(self, save_directory: str) -> None:
        json.dump(self.config, open(f"{save_directory}/encoder_config.json", "w"), indent=2, ensure_ascii=False)
        self.encoder.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, load_directory: str) -> "TextEmbeddingModel":
        config = json.load(open(f"{load_directory}/encoder_config.json", "r", encoding="utf-8"))
        model = cls(**config)
        model.encoder = load_model(load_directory)
        return model


class DualBERTModel(SpanEncoderModelBase):
    def __init__(self, model_name_or_path: str, pooling: str = "first", distance: str = "inner_product", temperature: float = 1.0) -> None:
        """Dual-Encoder BERT Model for Span Retrieval
        Args:
            model_name_or_path (str): Pre-trained model name or path
            pooling (str): Pooling method for obtaining fixed-size representations ('first', 'mean', "last")
            distance (str): Similarity metric for retrieval ('inner_product', 'cosine', 'euclidean')
            temperature (float): Temperature parameter for scaling similarity scores
        """
        super().__init__(model_name_or_path, pooling, distance, temperature)
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

    def use_blink_weights(self) -> None:
        if self.config["model_name"] != "google-bert/bert-large_uncased":
            raise ValueError("BLINK weights are only available for 'bert-large-uncased' model.")
        url = "http://dl.fbaipublicfiles.com/BLINK/biencoder_wiki_large.bin"
        state_dict = torch.load(io.BytesIO(requests.get(url).content))
        ctxt_dict, cand_dict = OrderedDict(), OrderedDict()
        for key, value in state_dict.items():
            if key[:26] == "context_encoder.bert_model":
                new_k = key[27:]
                ctxt_dict[new_k] = value
            if key[:23] == "cand_encoder.bert_model":
                new_k = key[24:]
                cand_dict[new_k] = value
        del state_dict
        self.mention_encoder.load_state_dict(ctxt_dict)
        self.candidate_encoder.load_state_dict(cand_dict)

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
            hard_negatives_input_ids: Optional[torch.Tensor] = None,
            hard_negatives_attention_mask: Optional[torch.Tensor] = None,
            hard_negatives_token_type_ids: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
        ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        mention_embeddings = self.encode_mention(input_ids, attention_mask, token_type_ids)
        candidates_embeddings = self.encode_candidate(candidates_input_ids, candidates_attention_mask, candidates_token_type_ids)

        bs, hs = candidates_embeddings.size(0), candidates_embeddings.size(-1)
        candidates_embeddings = candidates_embeddings.unsqueeze(0).repeat(bs, 1, 1)

        if hard_negatives_input_ids is not None and hard_negatives_attention_mask is not None:
            hard_negatives_embeddings = self.encode_mention(hard_negatives_input_ids, hard_negatives_attention_mask, hard_negatives_token_type_ids)
            hard_negatives_embeddings = hard_negatives_embeddings.view([bs, -1, hs])
            candidates_embeddings = torch.cat([candidates_embeddings, hard_negatives_embeddings], dim=1)

        if self.distance == 'inner_product':
            scores = torch.bmm(mention_embeddings.unsqueeze(1), candidates_embeddings.transpose(1, -1)).squeeze(1)
        elif self.distance == 'cosine':
            mention_embeddings_norm = mention_embeddings.unsqueeze(1) / torch.norm(mention_embeddings.unsqueeze(1), dim=2, keepdim=True)
            candidates_embeddings_norm = candidates_embeddings / torch.norm(candidates_embeddings, dim=2, keepdim=True)
            scores = torch.bmm(mention_embeddings_norm, candidates_embeddings_norm.transpose(1, -1)).squeeze(1)
        else:
            scores = torch.cdist(mention_embeddings.unsqueeze(1), candidates_embeddings).squeeze(1)

        if labels is None:
            return (None, scores)

        loss = torch.nn.functional.cross_entropy(scores, labels, reduction="mean")
        return (loss, scores)

    def resize_token_embeddings(self, new_num_tokens: int) -> None:
        self.mention_encoder.resize_token_embeddings(new_num_tokens)
        self.candidate_encoder.resize_token_embeddings(new_num_tokens)

    def save_pretrained(self, save_directory: str) -> None:
        json.dump(self.config, open(f"{save_directory}/encoder_config.json", "w"), indent=2, ensure_ascii=False)
        self.mention_encoder.save_pretrained(f"{save_directory}/context_model")
        self.candidate_encoder.save_pretrained(f"{save_directory}/candidate_model")

    @classmethod
    def from_pretrained(cls, load_directory: str) -> "DualBERTModel":
        config = json.load(open(f"{load_directory}/encoder_config.json", "r", encoding="utf-8"))
        config["model_name_or_path"] = config["model_name_or_path"]
        model = cls(**config)
        print(load_directory)
        print(config)
        model.mention_encoder = load_model(f"{load_directory}/context_model")
        model.candidate_encoder = load_model(f"{load_directory}/candidate_model")
        return model

# import json
import json
import os
from dataclasses import dataclass
from logging import getLogger
from typing import Optional, Union

import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerBase

from entity_linkings.data_utils import CollatorBase, EntityDictionary
from entity_linkings.models.span_retrieval.span_encoder import SpanEncoderModelBase

from ..base import IndexerBase

logger = getLogger(__name__)


class DenseRetriever(IndexerBase):
    @dataclass
    class Config(IndexerBase.Config):
        tokenizer: Optional[PreTrainedTokenizerBase] = None
        model: SpanEncoderModelBase = None
        batch_size: int = 16
        device: Optional[Union[str, torch.device]] = None
        metric: str = "cosine"
        use_hnsw: bool = False
        n_hubs: int = 10

    def __init__(self, dictionary: EntityDictionary, config: Optional[Config] = None) -> None:
        super().__init__(dictionary, config)
        if self.config.model is None:
            raise ValueError("Model must be provided.")
        if self.config.tokenizer is None:
            raise ValueError("Tokenizer must be provided.")
        self.model = self.config.model
        self.tokenizer = self.config.tokenizer
        self.vector_size = self.model.hidden_size

    def _initialize(self) -> None:
        self.meta_ids_to_keys: dict[int, str] = {}
        if self.config.metric not in ["cosine", "inner_product", "euclidean"]:
            raise NotImplementedError(f"{self.config.metric} is not supported")
        if self.config.use_hnsw:
            if self.config.metric == 'cosine' or self.config.metric == 'inner_product':
                self.index = faiss.IndexHNSWFlat(self.vector_size, self.config.n_hubs, faiss.METRIC_INNER_PRODUCT)
            else:
                self.index = faiss.IndexHNSWFlat(self.vector_size, self.config.n_hubs, faiss.METRIC_L2)
        else:
            if self.config.metric == 'cosine' or self.config.metric == 'inner_product':
                self.index = faiss.IndexFlatIP(self.vector_size)
            else:
                self.index = faiss.IndexFlatL2(self.vector_size)

    @torch.no_grad()
    def build_index(self, index_path: Optional[str] = None) -> None:
        if index_path and os.path.exists(os.path.join(index_path, "index.dpr")) and os.path.exists(os.path.join(index_path, "meta.json")):
            logger.info(f"Loading index from {index_path}")
            self.load(index_path)
        else:
            self._initialize()
            self.meta_ids_to_keys = {k: idx for idx, k in self.dictionary.id_to_index.items()}
            self.model.eval()
            self.model.to(self.config.device)

            dataloader = DataLoader(
                self.dictionary,
                collate_fn=CollatorBase(self.tokenizer),
                batch_size=self.config.batch_size,
                sampler=SequentialSampler(self.dictionary)
            )
            pbar = tqdm(total=len(dataloader), desc='Build Index')
            for batch in dataloader:
                pbar.update()
                batch = batch.to(self.config.device)
                entity_embedding = self.model.encode_candidate(**batch).to('cpu').detach().numpy().copy()
                if self.config.metric == 'cosine':
                    faiss.normalize_L2(entity_embedding)
                self.index.add(entity_embedding)
            pbar.close()
            if index_path:
                self.save_index(index_path)

    @torch.no_grad()
    def search_knn(self, query: str|list[str], top_k: int, ignore_ids: Optional[list[str]|list[list[str]]] = None) -> tuple[np.ndarray, list[list[str]]]:
        self.model.eval()
        self.model.to(self.config.device)
        model_inputs = self.tokenizer(query, padding=True, truncation=True, return_tensors="pt")
        model_inputs = model_inputs.to(self.config.device)
        query_embed = self.model.encode_mention(**model_inputs).to('cpu').detach().numpy().copy()
        if top_k <= 0:
            raise RuntimeError("K is zero or under zero.")
        if top_k > len(self.meta_ids_to_keys):
            top_k = len(self.meta_ids_to_keys)
            logger.warning(f"K is over size of dictionary. K is modified to size of dictionary to {len(self.meta_ids_to_keys)}")

        if self.config.metric == 'cosine':
            faiss.normalize_L2(query_embed)

        additional_top_k = 0
        if ignore_ids is not None:
            if isinstance(ignore_ids[0], list):
                additional_top_k = max([len(ids) for ids in ignore_ids])
            else:
                additional_top_k = len(ignore_ids)
        scores, results = self.index.search(query_embed, k=top_k+additional_top_k)

        indices_keys = []
        for i in range(len(results)):
            if ignore_ids is None:
                indices_keys.append([self.meta_ids_to_keys[ind] for ind in results[i]])
                continue
            candidate_ids = []
            for j in results[i]:
                key = self.meta_ids_to_keys[j]
                if key not in ignore_ids[i]:
                    candidate_ids.append(key)
            indices_keys.append(candidate_ids[:top_k])
        return scores, indices_keys

    def save_index(self, index_path: str, ensure_ascii: bool = False) -> None:
        logger.info("Serializing index to %s", index_path)
        if not os.path.isdir(index_path):
            os.makedirs(index_path, exist_ok=True)
        index_file = os.path.join(index_path, "index.dpr")
        meta_file = os.path.join(index_path, "meta.json")
        faiss.write_index(self.index, index_file)
        json.dump(self.meta_ids_to_keys, open(meta_file, 'w'), ensure_ascii=ensure_ascii)

    def load(self, index_path: str) -> None:
        self._initialize()
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"{index_path} is not found.")
        index_file = os.path.join(index_path, "index.dpr")
        meta_file = os.path.join(index_path, "meta.json")
        logger.info("Loading index from %s", index_file)
        self.index = faiss.read_index(index_file)
        self.meta_ids_to_keys.update({int(k): v for k, v in json.load(open(meta_file)).items()})
        logger.info(
            "Loaded index of type %s and size %d", type(self.index), len(self.meta_ids_to_keys)
        )

    def __len__(self) -> int:
        return len(self.meta_ids_to_keys)

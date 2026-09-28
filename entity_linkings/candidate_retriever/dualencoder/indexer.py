# import json
import json
import os
from logging import getLogger
from typing import Optional

import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm.auto import tqdm
from transformers import BatchEncoding, PreTrainedTokenizer

from entity_linkings.data_utils import CollatorBase, EntityDictionary

from ..base import IndexerBase
from .encoder import DualBERTModel

logger = getLogger(__name__)


class FaissIndexer(IndexerBase):
    def __init__(
        self,
        model: DualBERTModel,
        tokenizer: PreTrainedTokenizer,
        dictionary: EntityDictionary,
        batch_size: int = 16,
        metric: str = "cosine",
        use_hnsw: bool = False,
        n_hubs: int = 10,
        fp16: bool = False
    ) -> None:
        super().__init__(dictionary)
        self.model = model
        self.tokenizer = tokenizer
        self.vector_size = self.model.hidden_size
        self.metric = metric
        self.use_hnsw = use_hnsw
        self.n_hubs = n_hubs
        self.fp16 = fp16
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.batch_size = batch_size
        self.collator = CollatorBase(tokenizer)

    def _initialize(self) -> None:
        self.meta_ids_to_keys: dict[int, str] = {}
        if self.metric not in ["cosine", "inner_product", "euclidean"]:
            raise NotImplementedError(f"{self.metric} is not supported")
        if self.use_hnsw:
            if self.metric == 'cosine' or self.metric == 'inner_product':
                self.index = faiss.IndexHNSWFlat(self.vector_size, self.n_hubs, faiss.METRIC_INNER_PRODUCT)
            else:
                self.index = faiss.IndexHNSWFlat(self.vector_size, self.n_hubs, faiss.METRIC_L2)
        else:
            if self.metric == 'cosine' or self.metric == 'inner_product':
                self.index = faiss.IndexFlatIP(self.vector_size)
            else:
                self.index = faiss.IndexFlatL2(self.vector_size)

    @torch.no_grad()
    def build_index(self, index_path: str | None = None) -> None:
        if index_path is not None and os.path.exists(os.path.join(index_path, "index.dpr")) and os.path.exists(os.path.join(index_path, "meta.json")):
            logger.info(f"Loading index from {index_path}")
            self.load(index_path)
        else:
            if index_path is None:
                logger.warning("Index path is not provided. The index will not be saved after building. Consider providing an index path to save the built index for future use.")
            self._initialize()
            self.meta_ids_to_keys = {k: idx for idx, k in self.dictionary.id_to_index.items()}
            self.model.eval()
            self.model.to(self.device)

            dataloader = DataLoader(
                self.dictionary,
                collate_fn=self.collator,
                batch_size=self.batch_size,
                sampler=SequentialSampler(self.dictionary)
            )
            pbar = tqdm(total=len(dataloader), desc='Build Index')
            for batch in dataloader:
                pbar.update()
                batch = batch.to(self.device)
                if self.fp16:
                    with torch.autocast(device_type=self.device.type):
                        entity_embedding = self.model.encode_candidate(**batch).to('cpu').detach().numpy().copy()
                else:
                    entity_embedding = self.model.encode_candidate(**batch).to('cpu').detach().numpy().copy()
                if self.metric == 'cosine':
                    faiss.normalize_L2(entity_embedding)
                self.index.add(entity_embedding)
            pbar.close()

    def prepare_query(self, query: str|list[str]|BatchEncoding|list[BatchEncoding]) -> BatchEncoding:
        if isinstance(query, str) or (isinstance(query, list) and query and isinstance(query[0], str)):
            # The tokenizer pads and batches on its own. Passing that on to the collator
            # would wrap the batch in a second one and give the model a 3D input.
            return self.tokenizer(query, padding=True, truncation=True, return_tensors="pt")
        features = query if isinstance(query, list) else [query]
        return self.collator(features)

    @torch.no_grad()
    def search_knn(self, query: str|list[str]|BatchEncoding|list[BatchEncoding], top_k: int, ignore_ids: Optional[list[str]|list[list[str]]] = None) -> tuple[np.ndarray, list[list[str]]]:
        self.model.eval()
        self.model.to(self.device)
        model_inputs = self.prepare_query(query)
        model_inputs = model_inputs.to(self.device)
        if self.fp16:
            with torch.autocast(device_type=self.device.type):
                query_embed = self.model.encode_mention(**model_inputs).to('cpu').numpy()
        else:
            query_embed = self.model.encode_mention(**model_inputs).to('cpu').numpy()
        if top_k <= 0:
            raise RuntimeError("K is zero or under zero.")

        if self.metric == 'cosine':
            faiss.normalize_L2(query_embed)

        unwanted = self.normalize_ignore_ids(ignore_ids, len(query_embed))
        additional_top_k = max((len(ids) for ids in unwanted), default=0)

        # The ignored ids are dropped after retrieval, so that many hits are fetched on
        # top of the K that are wanted. Faiss pads the labels with -1 once K passes the
        # number of vectors it holds, and -1 maps to no entity, so cap K by what is left
        # once the ignored ids are taken out.
        index_size = len(self.meta_ids_to_keys)
        selectable = max(index_size - additional_top_k, 0)
        if top_k > selectable:
            logger.warning(f"K is over the number of selectable entities. K is modified to {selectable}.")
            top_k = selectable
        fetch_k = min(top_k + additional_top_k, index_size)
        if fetch_k == 0:
            return np.zeros((len(query_embed), 0)), [[] for _ in range(len(query_embed))]

        raw_scores, results = self.index.search(query_embed, k=fetch_k)

        # Each row is cut back to top_k, so the scores have to be cut the same way or
        # they stop lining up with the ids they belong to.
        kept_scores, indices_keys = [], []
        for i in range(len(results)):
            row_scores, candidate_ids = [], []
            for rank, j in enumerate(results[i]):
                key = self.meta_ids_to_keys[j]
                if key in unwanted[i]:
                    continue
                candidate_ids.append(key)
                row_scores.append(raw_scores[i][rank])
                if len(candidate_ids) == top_k:
                    break
            indices_keys.append(candidate_ids)
            kept_scores.append(row_scores)
        return np.array(kept_scores), indices_keys

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

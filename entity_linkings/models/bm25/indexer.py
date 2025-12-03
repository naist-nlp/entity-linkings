import json
import os
from dataclasses import dataclass
from logging import getLogger
from typing import Optional, Union

import bm25s
import numpy as np
from bm25s.hf import BM25HF
from bm25s.tokenization import Tokenized

from entity_linkings.entity_dictionary import EntityDictionaryBase

from ..base import IndexerBase
from .utils import ENGLISH_STOP_WORDS

logger = getLogger(__name__)


class BM25Indexer(IndexerBase):
    '''
    BM25 retriever using bm25s (https://github.com/xhluca/bm25s)
    '''

    @dataclass
    class Config(IndexerBase.Config):
        language: str = "en"
        n_threads: int = -1
        subword_tokenizer: Optional[str] = None

    def __init__(self, dictionary: EntityDictionaryBase, config: Optional[Config] = None) -> None:
        super().__init__(dictionary, config)
        if self.config.subword_tokenizer is not None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.subword_tokenizer)
            self.tokenize_func = self.subword_tokenize
        else:
            if self.config.language == "en":
                self.tokenize_func = self.whitespace_tokenize
            else:
                raise NotImplementedError(f"Language {self.config.language} is not supported yet.")

    def whitespace_tokenize(self, texts: Union[str, list[str]]) -> Tokenized:
        def _tokenize(text: list[str]) -> list[str]:
            return [token for token in text]

        corpus_token = bm25s.tokenize(texts, token_pattern=r"(?u)\b[\w#]+\b", stopwords=list(ENGLISH_STOP_WORDS), stemmer=_tokenize)
        return corpus_token

    def subword_tokenize(self, texts: Union[str, list[str]]) -> Tokenized:
        corpus_ids = []
        tokens_to_idx: dict[str, int] = {}
        for text in texts:
            tokens = self.tokenizer.tokenize(text, add_special_tokens=False)
            doc_ids = []
            for token in tokens:
                if token not in tokens_to_idx:
                    tokens_to_idx[token] = len(tokens_to_idx)
                token_id = tokens_to_idx[token]
                doc_ids.append(token_id)
            corpus_ids.append(doc_ids)
        return Tokenized(ids=corpus_ids, vocab=tokens_to_idx)

    def _initialize(self) -> None:
        self.index = BM25HF()
        self.meta_ids_to_keys: dict[int, str] = {}

    def build_index(self, index_path: Optional[str] = None) -> None:
        if index_path and os.path.exists(os.path.join(index_path, "meta_bm25.json")):
            self.load(index_path)
        else:
            self._initialize()
            descriptions = []
            for i, entity in enumerate(self.dictionary):
                self.meta_ids_to_keys[i] = entity["id"]
                descriptions.append(f"{entity['name']}: {entity['description']}")
            corpus_tokens = self.tokenize_func(descriptions)
            self.index.index(corpus_tokens)

    def search_knn(self, query: str|list[str], top_k: int, ignore_ids: Optional[list[str]|list[list[str]]] = None) -> tuple[np.ndarray, list[list[str]]]:
        if top_k <= 0:
            raise RuntimeError("K is zero or under zero.")
        if top_k > len(self.meta_ids_to_keys):
            top_k = len(self.meta_ids_to_keys)
            logger.warning(f"K is over size of dictionary. K is modified to size of dictionary to {len(self.meta_ids_to_keys)}")

        additional_top_k = 0
        if ignore_ids is not None:
            if isinstance(ignore_ids[0], list):
                additional_top_k = max([len(ids) for ids in ignore_ids])
            else:
                additional_top_k = len(ignore_ids)
        query_tokens = self.tokenize_func(query)
        results, scores = self.index.retrieve(
            query_tokens, k=top_k+additional_top_k, n_threads=self.config.n_threads
        )

        indices_keys = []
        for i in range(results.shape[0]):
            if not ignore_ids:
                indices_keys.append([self.meta_ids_to_keys[j] for j in results[i].tolist()])
                continue
            candidate_ids = []
            for j in results[i].tolist():
                key = self.meta_ids_to_keys[j]
                if key not in ignore_ids[i]:
                    candidate_ids.append(key)
            indices_keys.append(candidate_ids[:top_k])
        return scores, indices_keys

    def save_index(self, index_path: str, ensure_ascii: bool = False) -> None:
        logger.info("Serializing index to %s", index_path)
        self.index.save(index_path)
        meta_file = os.path.join(index_path, "meta_bm25.json")
        json.dump(self.meta_ids_to_keys, open(meta_file, 'w'), ensure_ascii=ensure_ascii)

    def load(self, index_path: str) -> None:
        self._initialize()
        logger.info("Deserializing index from %s", index_path)
        self.index = BM25HF.load(index_path, load_corpus=True)
        meta_file = os.path.join(index_path, "meta_bm25.json")
        self.meta_ids_to_keys.update({int(k): v for k, v in json.load(open(meta_file)).items()})
        logger.info(
            "Loaded index of type %s and size %d", type(self.index), len(self.meta_ids_to_keys)
        )

    def __len__(self) -> int:
        return len(self.meta_ids_to_keys)

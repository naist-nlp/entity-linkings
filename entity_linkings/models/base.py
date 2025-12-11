import abc
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer

from entity_linkings.data_utils import EntityDictionary
from entity_linkings.trainer import TrainingArguments


class IndexerBase(abc.ABC):
    @dataclass
    class Config: ...

    def __init__(self, dictionary: EntityDictionary, config: Optional[Config] = None) -> None:
        self.dictionary = dictionary
        self.config = config if config is not None else self.Config()

    def _initialize(self) -> None:
        raise NotImplementedError

    def build_index(self, index_path: Optional[str] = None) -> None:
        raise NotImplementedError

    def save_index(self, index_path: str, ensure_ascii: bool = False) -> None:
        raise NotImplementedError

    def load(self, index_path: str) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def search_knn(self, query: str|list[str], top_k: int, ignore_ids: Optional[list[str]|list[list[str]]] = None) -> tuple[np.ndarray, list[list[str]]]:
        raise NotImplementedError


class EntityRetrieverBase(abc.ABC):
    '''
    Base class for entity retrieval models
    '''
    @dataclass
    class Config:
        model_name_or_path: Optional[str] = None
        def to_dict(self) -> dict[str, Any]:
            return self.__dict__

    def __init__(self, dictionary: EntityDictionary, config: Optional[Config] = None) -> None:
        self.dictionary = dictionary
        self.config = config if config is not None else self.Config()

    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None, num_hard_negatives: int = 0, training_args: Optional[TrainingArguments] = None) -> dict[str, float]:
        raise NotImplementedError

    def evaluate(self, dataset: Dataset, **args: int) -> dict[str, float]:
        raise NotImplementedError

    def predict(self, sentence: str, spans: Optional[list[tuple[int, int]]] = None, top_k: int = 5) -> list[list[dict[str, Any]]]:
        raise NotImplementedError

    def retrieve_candidates(self, dataset: Dataset, top_k: int = 5, only_negative: bool = False, batch_size: int = 32, **args: int) -> list[list[str]]:
        raise NotImplementedError


class PipelineBase(abc.ABC):
    '''
    Base class for pipeline systems
    '''
    @dataclass
    class Config:
        model_name_or_path: str = "bert-base-uncased"
        task: str = "ed" # "ed" or "el"
        nil_token: str = "[NIL]"
        max_context_length: int = 128
        num_candidates: int = 30
        def to_dict(self) -> dict[str, Any]:
            return self.__dict__

    def __init__(self, retriever: EntityRetrieverBase, config: Optional[Config] = None) -> None:
        self.retriever = retriever
        self.dictionary = retriever.dictionary
        self.config = config if config is not None else self.Config()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name_or_path)
        self.tokenizer.add_special_tokens({"additional_special_tokens": [self.config.nil_token]})

    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None, num_candidates: int = 30, training_args: Optional[TrainingArguments] = None) -> dict[str, float]:
        raise NotImplementedError

    def evaluate(self, dataset: Dataset, num_candidates: int = 30, batch_size: int = 32, **args: int) -> dict[str, float]:
        raise NotImplementedError

    def predict(self, sentence: str, spans: Optional[list[tuple[int, int]]] = None, num_candidates: int = 30) -> list[list[dict[str, Any]]]:
        raise NotImplementedError

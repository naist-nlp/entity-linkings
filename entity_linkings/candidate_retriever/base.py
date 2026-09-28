import abc
from dataclasses import dataclass
from typing import Any, Optional, cast

import numpy as np
from datasets import Dataset
from transformers.trainer_utils import TrainOutput

from entity_linkings.data_utils import EntityDictionary
from entity_linkings.trainer import TrainingArguments
from entity_linkings.utils import BaseSystemOutput


class IndexerBase(abc.ABC):
    def __init__(self, dictionary: EntityDictionary) -> None:
        self.dictionary = dictionary
        self.entity_ids = self.dictionary.get_entity_ids()
        self.num_entities = len(self.dictionary)

    def _initialize(self) -> None:
        raise NotImplementedError

    def build_index(self, index_path: str | None = None) -> None:
        raise NotImplementedError

    def save_index(self, index_path: str, ensure_ascii: bool = False) -> None:
        raise NotImplementedError

    def load(self, index_path: str) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def search_knn(self, query: str|list[str], top_k: int, ignore_ids: Optional[list[str]|list[list[str]]] = None) -> tuple[np.ndarray, list[list[str]]]:
        raise NotImplementedError

    @staticmethod
    def normalize_ignore_ids(
            ignore_ids: Optional[list[str]|list[list[str]]],
            num_queries: int
        ) -> list[set[str]]:
        '''Turn either accepted shape into one set of ids per query.

        A flat list means the same ids are unwanted for every query. Indexing it by the
        query position instead, as the callers used to, picks out a single string and
        then tests the candidate against it as a substring.
        '''
        if not ignore_ids:
            return [set() for _ in range(num_queries)]
        if isinstance(ignore_ids[0], list):
            per_query = cast(list[list[str]], ignore_ids)
            if len(per_query) != num_queries:
                raise ValueError(
                    f"ignore_ids holds {len(per_query)} lists for {num_queries} queries; "
                    "pass one list per query, or a single flat list to apply to them all."
                )
            return [set(ids) for ids in per_query]
        shared = set(cast(list[str], ignore_ids))
        return [shared for _ in range(num_queries)]


class RetrieverBase(abc.ABC):
    '''
    Base class for entity retrieval models
    '''
    @dataclass
    class Config:
        model_name_or_path: Optional[str] = None
        def to_dict(self) -> dict[str, Any]:
            return self.__dict__

    def __init__(self, dictionary: EntityDictionary, config: Optional[Config] = None, index_path: Optional[str] = None) -> None:
        # index_path is part of the signature the CLIs call through get_retrievers, which
        # hands back a type[RetrieverBase]. Each subclass decides what to do with it.
        self.dictionary = dictionary
        self.config = config if config is not None else self.Config()

    def create_indexer(self, index_path: str | None = None) -> IndexerBase:
        raise NotImplementedError

    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None, num_hard_negatives: int = 0, training_args: Optional[TrainingArguments] = None) -> TrainOutput:
        raise NotImplementedError

    def evaluate(self, dataset: Dataset, **args: int) -> dict[str, float]:
        raise NotImplementedError

    def predict(self, sentence: str, spans: Optional[list[tuple[int, int]]] = None, top_k: int = 5) -> list[list[BaseSystemOutput]]:
        raise NotImplementedError

    def retrieve_candidates(self, dataset: Dataset, top_k: int = 5, only_negative: bool = False, batch_size: int = 32, **args: int) -> list[list[str]]:
        raise NotImplementedError

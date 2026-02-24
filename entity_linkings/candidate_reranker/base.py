import abc
from dataclasses import dataclass
from typing import Any, Optional

from datasets import Dataset

from entity_linkings.candidate_retriever import RetrieverBase
from entity_linkings.trainer import TrainingArguments
from entity_linkings.utils import BaseSystemOutput


class RerankerBase(abc.ABC):
    '''
    Base class for re-ranking models
    '''
    @dataclass
    class Config:
        model_name_or_path: Optional[str] = None
        def to_dict(self) -> dict[str, Any]:
            return self.__dict__

    def __init__(self, retriever: RetrieverBase, config: Optional[Config] = None) -> None:
        self.retriever = retriever
        self.dictionary = retriever.dictionary
        self.config = config if config is not None else self.Config()

    def train(
            self,
            train_dataset: Dataset,
            eval_dataset: Optional[Dataset] = None,
            num_candidates: int = 30,
            training_args: Optional[TrainingArguments] = None
        ) -> dict[str, float]:
        raise NotImplementedError

    def evaluate(self, dataset: Dataset, num_candidates: int = 30, batch_size: int = 32, **args: int) -> dict[str, float]:
        raise NotImplementedError

    def predict(self, sentence: str, spans: list[tuple[int, int]], num_candidates: int=30) -> list[BaseSystemOutput]:
        raise NotImplementedError

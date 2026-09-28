import logging
import random
from dataclasses import dataclass
from typing import Literal, Optional

from datasets import Dataset

from entity_linkings.data_utils import EntityDictionary
from entity_linkings.utils import BaseSystemOutput, calculate_recall_mrr

from ..base import RetrieverBase
from .indexer import BM25Indexer

logger = logging.getLogger(__name__)


class BM25(RetrieverBase):
    '''
    BM25 model for entity disambiguation
    '''
    @dataclass
    class Config(RetrieverBase.Config):
        language: str = "en"
        n_threads: int = -1
        subword_tokenizer: Optional[str] = None
        query_type_for_candidate: Literal['mention', 'description'] = 'mention'
        search_batch_size: int = 1024

    config: Config

    def __init__(self, dictionary: EntityDictionary, config: Optional[Config] = None, index_path: Optional[str] = None) -> None:
        super().__init__(dictionary, config)
        self.indexer = self.create_indexer(index_path=index_path) if index_path is not None else None

    def create_indexer(self, index_path: str | None = None) -> BM25Indexer:
        indexer =  BM25Indexer(
            dictionary=self.dictionary,
            language=self.config.language,
            n_threads=self.config.n_threads,
            subword_tokenizer=self.config.subword_tokenizer,
            query_type_for_candidate=self.config.query_type_for_candidate,
            batch_size=self.config.search_batch_size
        )
        indexer.build_index(index_path=index_path)
        return indexer

    def evaluate(self, dataset: Dataset, batch_size: int = 32, **args: int) -> dict[str, float]:
        if self.indexer is None:
            logger.warning("Indexer not found. Creating indexer with default settings. This may take some time if the index is large.")
            self.indexer = self.create_indexer(index_path=None)

        queries, labels = [], []
        for text, entities in zip(dataset["text"], dataset["entities"]):
            for ent in entities:
                ent_labels = ent['label']
                if not ent_labels:
                    continue
                if self.config.query_type_for_candidate == 'mention':
                    queries.append(text[ent["start"]: ent["end"]])
                else:
                    ent_label = random.choice(ent_labels)
                    queries.append(self.dictionary(ent_label)["description"])
                labels.append(ent_labels)

        _, all_indices = self.indexer.search_knn(queries, top_k=100, batch_size=batch_size)

        predictions = []
        for i, indices in enumerate(all_indices):
            preds = [{"id": self.dictionary(inds)["id"]} for inds in indices]
            predictions.append({"gold": labels[i], "predict": preds})
        metric = calculate_recall_mrr(predictions)
        return metric

    def predict(self, sentence: str, spans: Optional[list[tuple[int, int]]] = None, top_k: int = 5) -> list[list[BaseSystemOutput]]:
        if self.indexer is None:
            logger.warning("Indexer not found. Creating indexer with default settings. This may take some time if the index is large.")
            self.indexer = self.create_indexer(index_path=None)
        if not spans:
            raise ValueError("Spans must be provided for BM25 prediction.")

        queries = []
        for b, e in spans:
            queries.append(sentence[b:e])
        _, indices = self.indexer.search_knn(queries, top_k=top_k)
        all_result = []
        for i, (b, e) in enumerate(spans):
            result = []
            query = queries[i]
            for _, ind in enumerate(indices[i]):
                entry = self.dictionary(ind)
                result.append(BaseSystemOutput(query=query, start=b, end=e, id=entry['id']))
            all_result.append(result)
        return all_result

    def retrieve_candidates(self, dataset: Dataset, top_k: int = 5, only_negative: bool = False, batch_size: int = 32, **args: int) -> list[list[str]]:
        if self.indexer is None:
            logger.warning("Indexer not found. Creating indexer with default settings. This may take some time if the index is large.")
            self.indexer = self.create_indexer(index_path=None)

        queries, labels = [], []
        for example in dataset:
            text = example['text']
            for ent in example["entities"]:
                ent_labels = ent['label']
                if not ent_labels:
                    continue
                if self.config.query_type_for_candidate == 'mention':
                    queries.append(text[ent["start"]: ent["end"]])
                else:
                    ent_label = random.choice(ent_labels)
                    queries.append(self.dictionary(ent_label)["description"])
                labels.append(ent_labels)

        _, all_candidates = self.indexer.search_knn(
            queries,
            top_k=top_k,
            ignore_ids=labels if only_negative else None,
            batch_size=batch_size
        )
        return all_candidates

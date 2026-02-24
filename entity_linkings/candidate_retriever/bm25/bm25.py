import logging
import math
import random
from dataclasses import dataclass
from typing import Literal, Optional

from datasets import Dataset
from tqdm.auto import tqdm

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

    def __init__(self, dictionary: EntityDictionary, config: Optional[Config] = None, index_path: Optional[str] = None) -> None:
        super().__init__(dictionary, config)
        self.retriever = self.create_retriever(index_path=index_path)

    def create_retriever(self, index_path: Optional[str] = None) -> BM25Indexer:
        retriever =  BM25Indexer(
            dictionary=self.dictionary,
            language=self.config.language,
            n_threads=self.config.n_threads,
            subword_tokenizer=self.config.subword_tokenizer,
            query_type_for_candidate=self.config.query_type_for_candidate
        )
        retriever.build_index(index_path=index_path)
        return retriever

    def evaluate(self, dataset: Dataset, batch_size: int = 32, **args: int) -> dict[str, float]:
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

        predictions = []
        pbar = tqdm(total=(math.ceil(len(queries)/batch_size)), desc='Evaluate')
        for i in range(0, len(queries), batch_size):
            pbar.update()
            _, batch_indices = self.retriever.search_knn(queries[i:i + batch_size], top_k=100)
            batch_labels = labels[i:i + batch_size]
            for j, indices in enumerate(batch_indices):
                preds = [{"id": self.dictionary(inds)["id"]} for inds in indices]
                predictions.append({"gold": batch_labels[j], "predict": preds})
        pbar.close()
        metric = calculate_recall_mrr(predictions)
        return metric

    def predict(self, sentence: str, spans: Optional[list[tuple[int, int]]] = None, top_k: int = 5) -> list[list[BaseSystemOutput]]:
        if not spans:
            raise ValueError("Spans must be provided for BM25 prediction.")

        queries = []
        for b, e in spans:
            queries.append(sentence[b:e])
        similarities, indices = self.retriever.search_knn(queries, top_k=top_k)
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

        all_candidates = []
        pbar = tqdm(total=(math.ceil(len(queries)/batch_size)), desc='Retrieve candidates')
        for i in range(0, len(queries), batch_size):
            pbar.update()
            batch_queries = queries[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]
            _, batch_indices = self.retriever.search_knn(batch_queries, top_k=top_k, ignore_ids=batch_labels if only_negative else None)
            all_candidates.extend(batch_indices)
        pbar.close()

        return all_candidates

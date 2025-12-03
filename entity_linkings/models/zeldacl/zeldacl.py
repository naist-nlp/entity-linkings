import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

from datasets import Dataset
from tqdm.auto import tqdm

from entity_linkings.entity_dictionary import EntityDictionaryBase
from entity_linkings.utils import calculate_recall_mrr

from ..base import EntityRetrieverBase
from .indexer import ZeldaCandidateIndexer

logger = logging.getLogger(__name__)


class ZELDACL(EntityRetrieverBase):
    '''
    BM25 model for entity disambiguation
    '''
    @dataclass
    class Config(EntityRetrieverBase.Config): ...

    def __init__(self, dictionary: EntityDictionaryBase, config: Optional[Config] = None) -> None:
        super().__init__(dictionary, config)
        self.retriever =  ZeldaCandidateIndexer(dictionary=self.dictionary, config = ZeldaCandidateIndexer.Config())
        self.retriever.build_index(self.config.model_name_or_path)
        if self.config.model_name_or_path and not os.path.exists(self.config.model_name_or_path):
            self.retriever.save_index(self.config.model_name_or_path)
        if not self.config.model_name_or_path:
            logger.warning("model_name_or_path is not provided. The index will not be saved or loaded.")

    def evaluate(self, dataset: Dataset, **args: int) -> dict[str, float]:
        pbar = tqdm(total=(len(dataset)), desc='Predict')
        predictions = []
        for example in dataset:
            pbar.update()
            text = example['text']
            queries, labels = [], []
            for ent in example["entities"]:
                ent_labels = [label for label in ent['label']]
                if not ent_labels:
                    continue
                queries.append(text[ent["start"]: ent["end"]])
                labels.append(ent_labels)
            if len(queries) == 0:
                continue
            _, batch_indices = self.retriever.search_knn(queries, top_k=100)
            for i, indices in enumerate(batch_indices):
                preds = [{"id": self.dictionary(inds)["id"]} for inds in indices]
                predictions.append({"gold": labels[i], "predict": preds})
        pbar.close()
        metric = calculate_recall_mrr(predictions)
        return metric

    def predict(self, sentence: str, spans: Optional[list[tuple[int, int]]] = None, top_k: int = 5) -> list[list[dict[str, Any]]]:
        if not spans:
            raise ValueError("Spans must be provided for ZELDACL prediction.")

        queries = []
        for b, e in spans:
            queries.append(sentence[b:e])
        similarities, indices = self.retriever.search_knn(queries, top_k=top_k)
        all_result = []
        for i, (b, e) in enumerate(spans):
            result = []
            query = queries[i]
            for j, ind in enumerate(indices[i]):
                entry = self.dictionary(ind)
                result.append({
                    "query": query,
                    "prediction": entry['name'],
                    "id": entry['id'],
                    "description": entry['description'],
                    "score": similarities[i][j]
                })
            all_result.append(result)
        return all_result

    def retrieve_candidates(self, dataset: Dataset, top_k: int = 5, only_negative: bool = False, batch_size: int = 32, **args: int) -> list[list[str]]:
        all_candidates, queries, labels = [], [], []
        pbar = tqdm(total=(len(dataset)), desc='Retrieve candidates')
        for example in dataset:
            pbar.update()
            text = example['text']
            for ent in example["entities"]:
                queries.append(text[ent["start"]: ent["end"]])
                labels.append(ent['label'])
            if len(queries) >= batch_size:
                _, batch_indices = self.retriever.search_knn(queries, top_k=top_k, ignore_ids=labels if only_negative else None)
                all_candidates.extend(batch_indices)
                queries, labels = [], []
        if len(queries) > 0:
            _, batch_indices = self.retriever.search_knn(queries, top_k=top_k, ignore_ids=labels if only_negative else None)
            all_candidates.extend(batch_indices)
        pbar.close()
        return all_candidates

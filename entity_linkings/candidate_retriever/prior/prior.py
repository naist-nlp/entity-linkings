import logging
from dataclasses import dataclass
from typing import Optional

from datasets import Dataset
from tqdm.auto import tqdm

from entity_linkings.data_utils import EntityDictionary
from entity_linkings.utils import BaseSystemOutput, calculate_recall_mrr

from ..base import RetrieverBase
from .indexer import MentionPriorIndexer

logger = logging.getLogger(__name__)


class PRIOR(RetrieverBase):
    '''
    Prior probability model for entity disambiguation
    '''
    @dataclass
    class Config(RetrieverBase.Config): ...

    def __init__(self, dictionary: EntityDictionary, config: Optional[Config] = None, index_path: Optional[str] = None) -> None:
        super().__init__(dictionary, config)
        self.retriever = self.create_retriever(index_path=index_path)

    def create_retriever(self, index_path: Optional[str] = None) -> MentionPriorIndexer:
        retriever = MentionPriorIndexer(dictionary=self.dictionary, mention_counter_path=self.config.model_name_or_path)
        retriever.build_index(index_path=index_path)
        return retriever

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

    def predict(self, sentence: str, spans: Optional[list[tuple[int, int]]] = None, top_k: int = 5) -> list[list[BaseSystemOutput]]:
        if not spans:
            raise ValueError("Spans must be provided for PRIOR prediction.")

        queries = []
        for b, e in spans:
            queries.append(sentence[b:e])
        _, indices = self.retriever.search_knn(queries, top_k=top_k)
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

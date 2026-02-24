import abc
from typing import Optional

import spacy
from datasets import Dataset
from tqdm.auto import tqdm

from .candidate_reranker import RerankerBase
from .candidate_retriever import RetrieverBase
from .utils import BaseSystemOutput, calculate_inkb_f1


class ELPipeline(abc.ABC):
    def __init__(self, model: RetrieverBase | RerankerBase) -> None:
        self.model = model
        self.nlp = spacy.load("en_core_web_sm")

    def ner_predict(self, sentence: str) -> list[tuple[int, int]]:
        doc = self.nlp(sentence)
        spans = [(ent.start_char, ent.end_char) for ent in doc.ents]
        return spans

    def evaluate(self, dataset: Dataset, num_candidates: int = 30) -> dict[str, float]:
        pbar = tqdm(total=len(dataset), desc='Evaluate')
        all_predictions, all_golds = [], []
        for data in dataset:
            pbar.update()
            all_golds.append([ent for ent in data['entities'] if ent['label'] != ['-1']])
            # Mention detection
            text = data['text']
            pred_spans = self.ner_predict(text)
            if not pred_spans:
                all_predictions.append([])
                continue
            predictions = self.predict(text, spans=pred_spans, num_candidates=num_candidates)
            all_predictions.append([{'start': pred.start, 'end': pred.end, 'label': [pred.id]} for pred in predictions if pred.id != '-1'])
        pbar.close()
        metric = calculate_inkb_f1(all_predictions, all_golds)
        return metric

    def predict(self, sentence: str, spans: Optional[list[tuple[int, int]]] = None, num_candidates: int = 30) -> list[BaseSystemOutput]:
        if not spans:
            spans = self.ner_predict(sentence)
            if not spans:
                raise ValueError("No spans found using SpaCy. Please ensure the sentence contains named entities.")
        if isinstance(self.model, RetrieverBase):
            predictions = self.model.predict(sentence, spans, top_k=num_candidates)
            top1_predictions = [preds[0] for preds in predictions]
        elif isinstance(self.model, RerankerBase):
            top1_predictions = self.model.predict(sentence, spans, num_candidates=num_candidates)
        else:
            raise ValueError("The model should be either a RetrieverBase or RerankerBase instance.")
        return top1_predictions

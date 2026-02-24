from importlib.resources import files

import pytest
from datasets import load_dataset

import assets as test_data
from entity_linkings import ELPipeline, get_rerankers, get_retrievers, load_dictionary
from entity_linkings.utils import BaseSystemOutput

from .candidate_reranker import RerankerBase
from .candidate_retriever import RetrieverBase

dataset_path = str(files(test_data).joinpath("dataset_toy.jsonl"))
dictionary_path = str(files(test_data).joinpath("dictionary_toy.jsonl"))
dictionary = load_dictionary(dictionary_path)
dataset = load_dataset("json", data_files={"test": dataset_path})['test']

retriever_cls = get_retrievers("bm25")
retriever = retriever_cls(dictionary=dictionary)
reranker_cls = get_rerankers("crossencoder")
reranker = reranker_cls(retriever=retriever)

class TestELPipeline:
    @pytest.mark.parametrize("model", [reranker, retriever])
    def test__init__(self, model: RerankerBase | RetrieverBase) -> None:
        pipeline = ELPipeline(model=model)
        assert isinstance(pipeline, ELPipeline)
        assert pipeline.model == model
        assert hasattr(pipeline, "nlp")

    def test_ner_predict(self) -> None:
        pipeline = ELPipeline(model=retriever)
        sentence = "Steve Jobs was founder of Apple."
        spans = pipeline.ner_predict(sentence)
        assert isinstance(spans, list)
        assert spans == [(0, 10), (26, 31)]

    @pytest.mark.parametrize("model", [reranker, retriever])
    def test_predict(self, model: RerankerBase | RetrieverBase) -> None:
        pipeline = ELPipeline(model=model)
        sentence = "Steve Jobs was founder of Apple."
        spans = [(0, 10), (26, 31)]
        predictions = pipeline.predict(sentence, spans=spans, num_candidates=5)
        assert isinstance(predictions, list)
        for pred in predictions:
            assert isinstance(pred, BaseSystemOutput)
            assert pred.id is not None and pred.start is not None and pred.end is not None

    @pytest.mark.parametrize("model", [reranker, retriever])
    def test_evaluate(self, model: RerankerBase | RetrieverBase) -> None:
        pipeline = ELPipeline(model=model)
        metric = pipeline.evaluate(dataset, num_candidates=5)
        assert isinstance(metric, dict)
        assert 'precision' in metric and 'recall' in metric and 'f1' in metric
        assert 0.0 <= metric['precision'] <= 1.0
        assert 0.0 <= metric['recall'] <= 1.0
        assert 0.0 <= metric['f1'] <= 1.0

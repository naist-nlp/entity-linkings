from importlib.resources import files
from typing import Literal

import pytest

import assets as test_data
from entity_linkings import load_dataset, load_dictionary

from .bm25 import BM25

dataset_path = str(files(test_data).joinpath("dataset_toy.jsonl"))
dictionary_path = str(files(test_data).joinpath("dictionary_toy.jsonl"))
dictionary = load_dictionary(dictionary_path)
dataset = load_dataset(data_files={"test": dataset_path})['test']


@pytest.mark.span_retrieval_bm25
class TestBM25Model:
    def test_config(self) -> None:
        config = BM25.Config(
            model_name_or_path="test_bm25_index",
            n_threads=4
        )
        assert config.model_name_or_path == "test_bm25_index"
        assert config.n_threads == 4

    def test_init(self) -> None:
        bm25_model = BM25(
            dictionary=dictionary,
            config=BM25.Config(
                model_name_or_path="test_bm25_index",
                n_threads=4
            )
        )
        assert isinstance(bm25_model, BM25)
        assert bm25_model.retriever is not None
        assert bm25_model.dictionary is not None

    def test_evaluate(self) -> None:
        bm25_model = BM25(
            dictionary=dictionary,
            config=BM25.Config(
                model_name_or_path="test_bm25_index",
            )
        )
        metrics = bm25_model.evaluate(dataset)
        assert 'recall@1' in metrics
        assert 'recall@10' in metrics
        assert 'recall@50' in metrics
        assert 'recall@100' in metrics
        assert 'mrr@100' in metrics

    def test_predict(self) -> None:
        bm25_model = BM25(
            dictionary=dictionary,
            config=BM25.Config(
                model_name_or_path="test_bm25_index",
            )
        )
        sentence = "Steve Jobs was found Apple."
        spans = [(21, 26)]
        top_k = 3
        predictions = bm25_model.predict(sentence, spans, top_k)
        assert isinstance(predictions, list)
        assert len(predictions) == len(spans)
        for preds in predictions:
            assert isinstance(preds, list)
            assert len(preds) == min(top_k, len(dictionary))

    @pytest.mark.parametrize("query_type", ['mention', 'description'])
    def test_retrieve_candidates(self, query_type: Literal['mention', 'description']) -> None:
        bm25_model = BM25(
            dictionary=dictionary,
            config=BM25.Config(
                model_name_or_path="test_bm25_index",
                query_type_for_candidate=query_type
            )
        )
        top_k = 5
        candidates = bm25_model.retrieve_candidates(dataset, top_k=top_k, negative=False, batch_size=3)
        assert isinstance(candidates, list)
        total_entities = sum(len(example["entities"]) for example in dataset)
        assert len(candidates) == total_entities
        for inds in candidates:
            assert isinstance(inds, list)
            assert len(inds) == top_k

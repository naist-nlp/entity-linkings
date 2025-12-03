from importlib.resources import files

import numpy as np
import pytest
from bm25s.hf import BM25HF

import assets as test_data
from entity_linkings import load_dataset, load_dictionary

from .indexer import BM25Indexer

test_dataset_path = str(files(test_data).joinpath("dataset_toy.jsonl"))
dictionary_path = str(files(test_data).joinpath("dictionary_toy.jsonl"))
dictionary = load_dictionary(dictionary_path)
dataset = load_dataset("json", data_files={"test": test_dataset_path})["test"]

MODEL = ["google-bert/bert-base-uncased", None]

class TestBM25Indexer:
    @pytest.mark.parametrize("model_name", MODEL)
    def test___init__(self, model_name: str | None) -> None:
        retriever = BM25Indexer(dictionary, BM25Indexer.Config(subword_tokenizer=model_name))
        assert isinstance(retriever, BM25Indexer)
        assert retriever.dictionary == dictionary
        assert retriever.config.subword_tokenizer == model_name
        if model_name is None:
            assert retriever.tokenize_func == retriever.whitespace_tokenize
        else:
            assert retriever.tokenize_func == retriever.subword_tokenize

    @pytest.mark.parametrize("model_name", MODEL)
    def test_initialize(self, model_name: str | None) -> None:
        retriever = BM25Indexer(dictionary, BM25Indexer.Config(subword_tokenizer=model_name))
        retriever._initialize()
        assert isinstance(retriever.index, BM25HF)
        assert isinstance(retriever.meta_ids_to_keys, dict)
        assert len(retriever.meta_ids_to_keys) == 0

    @pytest.mark.parametrize("model_name", MODEL)
    def test_build_index(self, model_name: str | None) -> None:
        retriever = BM25Indexer(dictionary, BM25Indexer.Config(subword_tokenizer=model_name))
        retriever.build_index()
        assert len(retriever) == len(dictionary)
        assert len(list(retriever.meta_ids_to_keys.keys())) == len(dictionary)

    @pytest.mark.parametrize("top_k", [0, 2, 5, 10])
    def test_search_knn(self, top_k: int) -> None:
        retriever = BM25Indexer(dictionary)
        retriever.build_index()
        queries = ["Microsoft", "Apple"]
        if top_k <= 0:
            with pytest.raises(RuntimeError) as re:
                retriever.search_knn(queries, top_k)
            assert isinstance(re.value, RuntimeError)
            assert str(re.value) == "K is zero or under zero."
        else:
            distances, indices = retriever.search_knn(queries, top_k)
            assert isinstance(distances, np.ndarray) and isinstance(indices, list)
            if top_k > len(dictionary):
                assert distances.shape[0] == len(indices) == 2
                assert distances.shape[1] == len(indices[0]) == len(dictionary)
            else:
                assert distances.shape[0] == len(indices) == 2
                assert distances.shape[1] == len(indices[0]) == top_k

    @pytest.mark.parametrize("top_k", [2, 4])
    def test_search_knn_negatives(self, top_k: int) -> None:
        retriever = BM25Indexer(dictionary)
        retriever.build_index()
        for example in dataset:
            if not example["entities"]:
                continue
            ignore_ids = [entity["label"] for entity in example["entities"]]
            queries = [example["text"][ent["start"]: ent["end"]] for ent in example["entities"]]
            _, indices = retriever.search_knn(queries, top_k, ignore_ids=ignore_ids)
            for i, inds in enumerate(indices):
                assert len(inds) == top_k
                for ind in inds:
                    assert ind not in ignore_ids[i]

    def test_save_and_load(self) -> None:
        retriever = BM25Indexer(dictionary)
        retriever.build_index()
        retriever.save_index("bm25_test")

        loaded_retriever = BM25Indexer(dictionary, BM25Indexer.Config())
        loaded_retriever.build_index(index_path="bm25_test")
        assert retriever.dictionary and loaded_retriever.dictionary
        assert retriever.meta_ids_to_keys == loaded_retriever.meta_ids_to_keys

    def test_len(self) -> None:
        retriever = BM25Indexer(dictionary)
        retriever.build_index()
        assert len(retriever) == len(dictionary)

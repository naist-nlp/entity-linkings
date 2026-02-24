import tempfile
from importlib.resources import files

import numpy as np
import pytest
from bm25s.hf import BM25HF
from datasets import load_dataset

import assets as test_data
from entity_linkings import load_dictionary

from .indexer import BM25Indexer

dataset_path = str(files(test_data).joinpath("dataset_toy.jsonl"))
dictionary_path = str(files(test_data).joinpath("dictionary_toy.jsonl"))
dictionary = load_dictionary(dictionary_path)
dataset = load_dataset("json", data_files={"test": dataset_path})['test']

MODEL = ["google-bert/bert-base-uncased", None]

class TestBM25Indexer:
    @pytest.mark.parametrize("model_name", MODEL)
    def test___init__(self, model_name: str | None) -> None:
        indexer = BM25Indexer(dictionary, subword_tokenizer=model_name)
        assert isinstance(indexer, BM25Indexer)
        assert indexer.dictionary == dictionary
        if model_name is None:
            assert indexer.tokenize_func == indexer.whitespace_tokenize
        else:
            assert indexer.tokenize_func == indexer.subword_tokenize

    @pytest.mark.parametrize("model_name", MODEL)
    def test_initialize(self, model_name: str | None) -> None:
        indexer = BM25Indexer(dictionary, subword_tokenizer=model_name)
        indexer._initialize()
        assert isinstance(indexer.index, BM25HF)
        assert isinstance(indexer.meta_ids_to_keys, dict)
        assert len(indexer.meta_ids_to_keys) == 0

    @pytest.mark.parametrize("model_name", MODEL)
    def test_build_index(self, model_name: str | None) -> None:
        indexer = BM25Indexer(dictionary, subword_tokenizer=model_name)
        indexer.build_index()
        assert len(indexer) == len(dictionary)
        assert len(list(indexer.meta_ids_to_keys.keys())) == len(dictionary)

    @pytest.mark.parametrize("top_k", [0, 2, 5, 10])
    def test_search_knn(self, top_k: int) -> None:
        indexer = BM25Indexer(dictionary)
        indexer.build_index()
        queries = ["Microsoft", "Apple"]
        if top_k <= 0:
            with pytest.raises(RuntimeError) as re:
                indexer.search_knn(queries, top_k)
            assert isinstance(re.value, RuntimeError)
            assert str(re.value) == "K is zero or under zero."
        else:
            distances, indices = indexer.search_knn(queries, top_k)
            assert isinstance(distances, np.ndarray) and isinstance(indices, list)
            if top_k > len(dictionary):
                assert distances.shape[0] == len(indices) == 2
                assert distances.shape[1] == len(indices[0]) == len(dictionary)
            else:
                assert distances.shape[0] == len(indices) == 2
                assert distances.shape[1] == len(indices[0]) == top_k

    @pytest.mark.parametrize("top_k", [2, 4])
    def test_search_knn_negatives(self, top_k: int) -> None:
        indexer = BM25Indexer(dictionary)
        indexer.build_index()
        for example in dataset:
            if not example["entities"]:
                continue
            ignore_ids = [entity["label"] for entity in example["entities"]]
            queries = [example["text"][ent["start"]: ent["end"]] for ent in example["entities"]]
            _, indices = indexer.search_knn(queries, top_k, ignore_ids=ignore_ids)
            for i, inds in enumerate(indices):
                assert len(inds) == top_k
                for ind in inds:
                    assert ind not in ignore_ids[i]

    def test_save_and_load(self) -> None:
        indexer = BM25Indexer(dictionary)
        indexer.build_index()
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer.save_index(tmpdir)
            loaded_indexer = BM25Indexer(dictionary)
            loaded_indexer.build_index(index_path=tmpdir)
            assert indexer.dictionary and loaded_indexer.dictionary
            assert indexer.meta_ids_to_keys == loaded_indexer.meta_ids_to_keys

    def test_len(self) -> None:
        indexer = BM25Indexer(dictionary)
        indexer.build_index()
        assert len(indexer) == len(dictionary)

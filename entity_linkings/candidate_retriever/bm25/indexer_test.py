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

MODEL = ["hf-internal-testing/tiny-random-BertModel", None]

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

    @pytest.mark.parametrize("num_ignored", [1, len(dictionary) - 1, len(dictionary)])
    def test_search_knn_caps_k_by_what_is_left_after_ignoring(self, num_ignored: int) -> None:
        # The ignored ids are fetched on top of top_k, which used to push K past the
        # corpus size and make bm25s raise.
        indexer = BM25Indexer(dictionary)
        indexer.build_index()
        ignore_ids = [dictionary.get_entity_ids()[:num_ignored]]

        _, indices = indexer.search_knn(["Steve Jobs"], len(dictionary), ignore_ids=ignore_ids)

        assert len(indices[0]) == len(dictionary) - num_ignored
        assert not set(indices[0]) & set(ignore_ids[0])

    @pytest.mark.parametrize("ignore_ids", [["-1", "000015"], [["-1", "000015"]]])
    def test_search_knn_reads_both_shapes_of_ignore_ids(self, ignore_ids: list) -> None:
        # A flat list means the same ids are unwanted for every query. Indexing it by
        # query position instead picked out one string and matched candidates against
        # it as a substring, so everything past the first entry was let through.
        indexer = BM25Indexer(dictionary)
        indexer.build_index()

        scores, indices = indexer.search_knn(["Steve Jobs"], 4, ignore_ids=ignore_ids)

        assert not set(indices[0]) & {"-1", "000015"}
        assert scores.shape[1] == len(indices[0])

    def test_search_knn_rejects_a_mismatched_ignore_ids(self) -> None:
        indexer = BM25Indexer(dictionary)
        indexer.build_index()

        with pytest.raises(ValueError, match="one list per query"):
            indexer.search_knn(["Steve Jobs", "Apple"], 2, ignore_ids=[["-1"]])

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

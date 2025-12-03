from importlib.resources import files

import numpy as np
import pytest

import assets as test_data
from entity_linkings import load_dataset, load_dictionary

from .indexer import ZeldaCandidateIndexer

test_dataset_path = str(files(test_data).joinpath("dataset_toy.jsonl"))
dictionary_path = str(files(test_data).joinpath("dictionary_toy.jsonl"))
dictionary = load_dictionary(dictionary_path)
dataset = load_dataset("json", data_files={"test": test_dataset_path})["test"]

class TestZeldaCandidateIndexer:
    def test___init__(self) -> None:
        retriever = ZeldaCandidateIndexer(dictionary, ZeldaCandidateIndexer.Config())
        assert isinstance(retriever, ZeldaCandidateIndexer)
        assert retriever.dictionary == dictionary

    def test_build_index(self) -> None:
        retriever = ZeldaCandidateIndexer(dictionary)
        retriever.build_index()
        assert len(retriever) == len(dictionary)
        assert len(list(retriever.meta_ids_to_keys.keys())) == len(dictionary)

    def test_save_and_load(self) -> None:
        retriever = ZeldaCandidateIndexer(dictionary)
        retriever.build_index()
        retriever.save_index("zelda_candidate_test")

        loaded_retriever = ZeldaCandidateIndexer(dictionary)
        loaded_retriever.build_index("zelda_candidate_test")
        assert retriever.dictionary and loaded_retriever.dictionary
        assert retriever.meta_ids_to_keys == loaded_retriever.meta_ids_to_keys

    @pytest.mark.parametrize("top_k", [0, 2, 5, 10])
    def test_search_knn(self, top_k: int) -> None:
        retriever = ZeldaCandidateIndexer(dictionary)
        retriever.build_index("zelda_candidate_test")
        queries = ["Microsoft", "Meta", "nahanaha"]
        if top_k <= 0:
            with pytest.raises(RuntimeError) as re:
                retriever.search_knn(queries, top_k)
            assert isinstance(re.value, RuntimeError)
            assert str(re.value) == "K is zero or under zero."
        else:
            distances, indices = retriever.search_knn(queries, top_k)
            assert isinstance(distances, np.ndarray) and isinstance(indices, list)
            if top_k > len(dictionary):
                assert distances.shape[0] == len(indices) == 3
                assert distances.shape[1] == len(indices[0]) == len(dictionary)
            else:
                assert distances.shape[0] == len(indices) == 3
                assert distances.shape[1] == len(indices[0]) == top_k

    @pytest.mark.parametrize("top_k", [2, 4])
    def test_search_knn_negatives(self, top_k: int) -> None:
        retriever = ZeldaCandidateIndexer(dictionary)
        retriever.build_index("zelda_candidate_test")
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

    def test_len(self) -> None:
        retriever = ZeldaCandidateIndexer(dictionary)
        retriever.build_index("zelda_candidate_test")
        assert len(retriever) == len(dictionary)

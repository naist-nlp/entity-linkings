import tempfile
from importlib.resources import files

import numpy as np
import pytest
from datasets import load_dataset

import assets as test_data
from entity_linkings import load_dictionary

from .indexer import MentionPriorIndexer

dataset_path = str(files(test_data).joinpath("dataset_toy.jsonl"))
dictionary_path = str(files(test_data).joinpath("dictionary_toy.jsonl"))
mention_counter_path = str(files(test_data).joinpath("mention_counter_toy.json"))
dictionary = load_dictionary(dictionary_path)
dataset = load_dataset("json", data_files={"test": dataset_path})['test']


@pytest.fixture(scope='module')
def mention_prior_indexer() -> MentionPriorIndexer:
    indexer = MentionPriorIndexer(dictionary=dictionary, mention_counter_path=mention_counter_path)
    with tempfile.TemporaryDirectory() as tmpdir:
        indexer.build_index(tmpdir)
    return indexer

class TestMentionPriorIndexer:
    def test_build_index(self) -> None:
        indexer = MentionPriorIndexer(dictionary=dictionary, mention_counter_path=mention_counter_path)
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer.build_index(tmpdir)
            assert len(indexer) == len(dictionary)
            assert len(list(indexer.meta_ids_to_keys.keys())) == len(dictionary)

    @pytest.mark.parametrize("top_k", [0, 1, 2])
    def test_search_knn(self, top_k: int) -> None:
        queries = ["Microsoft", "Meta", "nahanaha"]
        if top_k <= 0:
            indexer = MentionPriorIndexer(dictionary=dictionary, mention_counter_path=mention_counter_path)
            with tempfile.TemporaryDirectory() as tmpdir:
                indexer.build_index(tmpdir)
                with pytest.raises(RuntimeError) as re:
                    indexer.search_knn(queries, top_k)
                assert isinstance(re.value, RuntimeError)
                assert str(re.value) == "K is zero or under zero."
        else:
            indexer = MentionPriorIndexer(dictionary=dictionary, mention_counter_path=mention_counter_path)
            with tempfile.TemporaryDirectory() as tmpdir:
                indexer.build_index(tmpdir)
                distances, indices = indexer.search_knn(queries, top_k)
                assert isinstance(distances, np.ndarray) and isinstance(indices, list)
                if top_k > len(dictionary):
                    assert distances.shape[0] == len(indices) == 3
                    assert distances.shape[1] == len(indices[0]) == len(dictionary)
                else:
                    assert distances.shape[0] == len(indices) == 3
                    assert distances.shape[1] == len(indices[0]) == top_k

    @pytest.mark.parametrize("top_k", [1])
    def test_search_knn_negatives(self, mention_prior_indexer: MentionPriorIndexer, top_k: int) -> None:
        for example in dataset:
            if not example["entities"]:
                continue
            ignore_ids = [entity["label"] for entity in example["entities"]]
            queries = [example["text"][ent["start"]: ent["end"]] for ent in example["entities"]]
            _, indices = mention_prior_indexer.search_knn(queries, top_k, ignore_ids=ignore_ids)
            for i, inds in enumerate(indices):
                assert len(inds) == top_k
                for ind in inds:
                    assert ind not in ignore_ids[i]

    def test_len(self, mention_prior_indexer: MentionPriorIndexer) -> None:
        assert len(mention_prior_indexer) == len(dictionary)

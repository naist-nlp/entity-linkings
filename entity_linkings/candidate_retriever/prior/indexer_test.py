import copy
import tempfile
from collections.abc import Callable
from importlib.resources import files

import numpy as np
import pytest
from datasets import load_dataset

import assets as test_data
from entity_linkings import load_dictionary

from .indexer import (
    MentionPriorIndexer,
    build_most_simpler_mentions_dict,
    build_simpler_mentions_dict,
)

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

@pytest.mark.parametrize(
    "build, expected_key",
    [(build_simpler_mentions_dict, "newyork"), (build_most_simpler_mentions_dict, "newyork")],
)
def test_build_dict_merges_without_touching_the_input(
    build: Callable[[dict[str, dict[str, int]]], dict[str, dict[str, int]]], expected_key: str
) -> None:
    # "New York", "newyork" and "New  York" all collapse to the same simplified key, so
    # their counts have to be summed into a bucket of their own. Reusing the caller's
    # dict as that bucket would push the other mentions' counts into mention_id_counter,
    # the table the exact-match lookup reads.
    counter = {"New York": {"Q60": 10}, "newyork": {"Q61": 5}, "New  York": {"Q60": 3}}
    unchanged = copy.deepcopy(counter)

    merged = build(counter)

    assert counter == unchanged
    assert merged[expected_key] == {"Q60": 13, "Q61": 5}
    assert all(bucket is not counter[mention] for mention, bucket in zip(counter, merged.values()))


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

    @pytest.mark.parametrize("num_ignored", [len(dictionary) - 1, len(dictionary)])
    def test_search_knn_caps_k_when_too_much_is_ignored(
        self, mention_prior_indexer: MentionPriorIndexer, num_ignored: int
    ) -> None:
        # Asking for more candidates than the dictionary can still supply once the
        # ignored ids are taken out used to spin forever in the padding loop, so a
        # regression here shows up as a hanging test rather than a failing one.
        ignore_ids = [dictionary.get_entity_ids()[:num_ignored]]
        top_k = len(dictionary)

        distances, indices = mention_prior_indexer.search_knn(["Microsoft"], top_k, ignore_ids=ignore_ids)

        selectable = len(dictionary) - num_ignored
        assert len(indices[0]) == selectable
        assert distances.shape == (1, selectable)
        assert not set(indices[0]) & set(ignore_ids[0])

    def test_len(self, mention_prior_indexer: MentionPriorIndexer) -> None:
        assert len(mention_prior_indexer) == len(dictionary)

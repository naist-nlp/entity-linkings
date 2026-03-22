import tempfile
from importlib.resources import files

import numpy as np
import pytest
from datasets import load_dataset
from faiss import (
    METRIC_INNER_PRODUCT,
    METRIC_L2,
    IndexFlatIP,
    IndexFlatL2,
    IndexHNSWFlat,
)
from transformers import AutoTokenizer

import assets as test_data
from entity_linkings import load_dictionary

from .encoder import DualBERTModel
from .indexer import FaissIndexer
from .preprocessor import DualEncoderPreprocessor

model_name = 'google-bert/bert-base-uncased'
model = DualBERTModel(model_name_or_path=model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
special_tokens = {"additional_special_tokens": ["[ENT]", "[START_ENT]", "[END_ENT]"]}
tokenizer.add_special_tokens(special_tokens)
model.resize_token_embeddings(len(tokenizer))

test_dataset_path = str(files(test_data).joinpath("dataset_toy.jsonl"))
dictionary_path = str(files(test_data).joinpath("dictionary_toy.jsonl"))
dictionary = load_dictionary(dictionary_path)
dataset = load_dataset("json", data_files={"test": test_dataset_path})['test']
preprocessor = DualEncoderPreprocessor(tokenizer, "[START_ENT]", "[END_ENT]", "[ENT]", 128, 128, 500)
processed_dictionary = preprocessor.dictionary_preprocess(dictionary)


class TestFaissIndexer:
    def test___init__(self) -> None:
        indexer = FaissIndexer(
            model = model,
            tokenizer=tokenizer,
            dictionary=processed_dictionary,
        )
        assert isinstance(indexer, FaissIndexer)
        assert indexer.model is model
        assert indexer.dictionary is processed_dictionary
        assert indexer.vector_size == model.hidden_size

    @pytest.mark.parametrize("metric",["inner_product", "cosine", "euclidean"])
    @pytest.mark.parametrize("use_hnsw",[True, False])
    def test_initialize(self, use_hnsw: bool, metric: str) -> None:
        indexer = FaissIndexer(
            model=model,
            tokenizer=tokenizer,
            dictionary=processed_dictionary,
            use_hnsw=use_hnsw,
            metric=metric,
        )
        indexer._initialize()
        if use_hnsw:
            if metric == "euclidean":
                assert isinstance(indexer.index, IndexHNSWFlat) and indexer.index.metric_type == METRIC_L2
            else:
                assert isinstance(indexer.index, IndexHNSWFlat) and indexer.index.metric_type == METRIC_INNER_PRODUCT
        else:
            if metric == "euclidean":
                assert isinstance(indexer.index, IndexFlatL2)
            else:
                assert isinstance(indexer.index, IndexFlatIP)
        assert hasattr(indexer, "meta_ids_to_keys") and isinstance(indexer.meta_ids_to_keys, dict)
        assert len(indexer.meta_ids_to_keys) == indexer.index.ntotal == 0

    def test_build_index(self) -> None:
        indexer = FaissIndexer(
            model=model,
            tokenizer=tokenizer,
            dictionary=processed_dictionary,
        )
        indexer.build_index()
        assert isinstance(indexer.index, IndexFlatIP)
        assert isinstance(indexer.meta_ids_to_keys, dict)
        assert len(indexer.meta_ids_to_keys) == indexer.index.ntotal == len(dictionary)

    @pytest.mark.parametrize("top_k", [0, 2, 5, 10])
    def test_search_knn(self, top_k: int) -> None:
        indexer = FaissIndexer(
            model=model,
            tokenizer=tokenizer,
            dictionary=processed_dictionary,
            batch_size=2,
        )
        indexer.build_index()
        for data in dataset:
            query = data['text']
            if top_k <= 0:
                with pytest.raises(RuntimeError) as re:
                    indexer.search_knn(query, top_k)
                assert isinstance(re.value, RuntimeError)
                assert str(re.value) == "K is zero or under zero."
                break
            else:
                distances, indices = indexer.search_knn(query, top_k=top_k)
                assert isinstance(distances, np.ndarray) and isinstance(indices, list)
                if top_k > len(dictionary):
                    assert distances.shape[0] == len(indices) == 1
                    assert distances.shape[1] == len(indices[0]) == len(dictionary)
                else:
                    assert distances.shape[0] == len(indices) == 1
                    assert distances.shape[1] == len(indices[0]) == top_k

    @pytest.mark.parametrize("top_k", [2, 4])
    def test_search_knn_negatives(self, top_k: int) -> None:
        indexer = FaissIndexer(
            model=model,
            tokenizer=tokenizer,
            dictionary=processed_dictionary,
        )
        indexer.build_index()
        for text, entities in zip(dataset['text'], dataset['entities']):
            for ent in entities:
                if 'label' not in ent:
                    continue
                query = text
                labels = ent["label"]
                _, indices = indexer.search_knn(query, top_k, ignore_ids=labels)
                for i, inds in enumerate(indices):
                    assert len(inds) == top_k
                    for ind in inds:
                        assert ind not in labels[i]

    def test_save_and_load(self) -> None:
        indexer = FaissIndexer(
            model=model,
            tokenizer=tokenizer,
            dictionary=processed_dictionary,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer.build_index(tmpdir)
            loaded_indexer = FaissIndexer(
                model=model,
                tokenizer=tokenizer,
                dictionary=processed_dictionary,
            )
            loaded_indexer.build_index(tmpdir)
            assert indexer.dictionary and loaded_indexer.dictionary
            assert indexer.meta_ids_to_keys == loaded_indexer.meta_ids_to_keys
            assert indexer.dictionary.id_to_index == loaded_indexer.dictionary.id_to_index

    def test_len(self) -> None:
        indexer = FaissIndexer(
            model=model,
            tokenizer=tokenizer,
            dictionary=processed_dictionary,
        )
        indexer.build_index()
        assert len(indexer) == len(dictionary)

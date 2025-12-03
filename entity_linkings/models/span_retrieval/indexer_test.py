from importlib.resources import files
from typing import Iterator

import numpy as np
import pytest
from datasets import Dataset, DatasetDict
from faiss import (
    METRIC_INNER_PRODUCT,
    METRIC_L2,
    IndexFlatIP,
    IndexFlatL2,
    IndexHNSWFlat,
)
from transformers import AutoTokenizer, BatchEncoding

import assets as test_data
from entity_linkings import load_dataset, load_dictionary
from entity_linkings.dataset.utils import preprocess
from entity_linkings.entity_dictionary.base import EntityDictionaryBase
from entity_linkings.models.span_retrieval import DualBERTModel

from .indexer import DenseRetriever

model_name = 'google-bert/bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
special_tokens = {"additional_special_tokens": ["[ENT]", "[START_ENT]", "[END_ENT]"]}
tokenizer.add_special_tokens(special_tokens)

test_dataset_path = str(files(test_data).joinpath("dataset_toy.jsonl"))
dictionary_path = str(files(test_data).joinpath("dictionary_toy.jsonl"))
dictionary = load_dictionary(dictionary_path)
dataset = load_dataset("json", data_files={"test": test_dataset_path})['test']

MODELS = [DualBERTModel(model_name_or_path=model_name)]

def dataset_preprocess(dataset: Dataset) -> DatasetDict:
    def _preprocess_example(examples: Dataset) -> Iterator[BatchEncoding]:
        for text, entities in zip(examples["text"], examples["entities"]):
            for ent in entities:
                text = text[:ent["start"]] + "[START_ENT]" + text[ent["start"]:ent["end"]] + "[END_ENT]" + text[ent["end"]:]
                encodings = tokenizer(
                    text,
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                encodings["labels"] = [dictionary.get_label_id(label) for label in ent["label"]]
                yield encodings
    return preprocess(dataset, _preprocess_example)


def dictionary_preprocess(dictionary: EntityDictionaryBase) -> EntityDictionaryBase:
    def preprocess_example(name: str, description: str) -> dict[str, list[int]]:
        text = name + " " + description
        encodings  = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
        )
        return encodings
    dictionary.add_encoding(preprocess_example)
    return dictionary


class TestDenseRetriever:
    @pytest.mark.parametrize("model", MODELS)
    def test___init__(self, model: DualBERTModel) -> None:
        processed_dictionary = dictionary_preprocess(dictionary)
        retriever = DenseRetriever(
            dictionary=processed_dictionary,
            config=DenseRetriever.Config(
                model=model,
                tokenizer=tokenizer,
            )
        )
        assert isinstance(retriever, DenseRetriever)
        assert isinstance(retriever.config, DenseRetriever.Config)
        assert retriever.model is model
        assert retriever.dictionary is processed_dictionary
        assert retriever.vector_size == model.hidden_size

    @pytest.mark.parametrize("model", MODELS)
    @pytest.mark.parametrize("metric",["inner_product", "cosine", "euclidean"])
    @pytest.mark.parametrize("use_hnsw",[True, False])
    def test_initialize(self, model: DualBERTModel, use_hnsw: bool, metric: str) -> None:
        processed_dictionary = dictionary_preprocess(dictionary)
        retriever = DenseRetriever(
            dictionary=processed_dictionary,
            config=DenseRetriever.Config(
                model=model,
                tokenizer=tokenizer,
                use_hnsw=use_hnsw,
                metric=metric,
            )
        )
        retriever._initialize()
        if use_hnsw:
            if metric == "euclidean":
                assert isinstance(retriever.index, IndexHNSWFlat) and retriever.index.metric_type == METRIC_L2
            else:
                assert isinstance(retriever.index, IndexHNSWFlat) and retriever.index.metric_type == METRIC_INNER_PRODUCT
        else:
            if metric == "euclidean":
                assert isinstance(retriever.index, IndexFlatL2)
            else:
                assert isinstance(retriever.index, IndexFlatIP)
        assert hasattr(retriever, "meta_ids_to_keys") and isinstance(retriever.meta_ids_to_keys, dict)
        assert len(retriever.meta_ids_to_keys) == retriever.index.ntotal == 0

    @pytest.mark.parametrize("model", MODELS)
    def test_build_index(self, model: DualBERTModel) -> None:
        model.resize_token_embeddings(len(tokenizer))
        processed_dictionary = dictionary_preprocess(dictionary)
        retriever = DenseRetriever(
            dictionary=processed_dictionary,
            config=DenseRetriever.Config(
                model=model,
                tokenizer=tokenizer,
            )
        )
        retriever.build_index()
        assert isinstance(retriever.index, IndexFlatIP)
        assert isinstance(retriever.meta_ids_to_keys, dict)
        assert len(retriever.meta_ids_to_keys) == retriever.index.ntotal == len(dictionary)

    @pytest.mark.parametrize("top_k", [0, 2, 5, 10])
    @pytest.mark.parametrize("model", MODELS)
    def test_search_knn(self, top_k: int, model: DualBERTModel) -> None:
        model.resize_token_embeddings(len(tokenizer))
        processed_dictionary = dictionary_preprocess(dictionary)
        retriever = DenseRetriever(
            dictionary=processed_dictionary,
            config=DenseRetriever.Config(
                model=model,
                tokenizer=tokenizer,
                batch_size=2,
            )
        )
        retriever.build_index()
        for data in dataset:
            query = data['text']
            if top_k <= 0:
                with pytest.raises(RuntimeError) as re:
                    retriever.search_knn(query, top_k)
                assert isinstance(re.value, RuntimeError)
                assert str(re.value) == "K is zero or under zero."
                break
            else:
                distances, indices = retriever.search_knn(query, top_k=top_k)
                assert isinstance(distances, np.ndarray) and isinstance(indices, list)
                if top_k > len(dictionary):
                    assert distances.shape[0] == len(indices) == 1
                    assert distances.shape[1] == len(indices[0]) == len(dictionary)
                else:
                    assert distances.shape[0] == len(indices) == 1
                    assert distances.shape[1] == len(indices[0]) == top_k

    @pytest.mark.parametrize("top_k", [2, 4])
    @pytest.mark.parametrize("model", MODELS)
    def test_search_knn_negatives(self, top_k: int, model: DualBERTModel) -> None:
        model.resize_token_embeddings(len(tokenizer))
        processed_dictionary = dictionary_preprocess(dictionary)
        retriever = DenseRetriever(
            dictionary=processed_dictionary,
            config=DenseRetriever.Config(
                model=model,
                tokenizer=tokenizer,
            )
        )
        retriever.build_index()
        for text, entities in zip(dataset['text'], dataset['entities']):
            for ent in entities:
                if 'label' not in ent:
                    continue
                query = text
                labels = ent["label"]
                _, indices = retriever.search_knn(query, top_k, ignore_ids=labels)
                for i, inds in enumerate(indices):
                    assert len(inds) == top_k
                    for ind in inds:
                        assert ind not in labels[i]

    @pytest.mark.parametrize("model", MODELS)
    def test_save_and_load(self, model: DualBERTModel) -> None:
        model.resize_token_embeddings(len(tokenizer))
        processed_dictionary = dictionary_preprocess(dictionary)
        retriever = DenseRetriever(
            dictionary=processed_dictionary,
            config=DenseRetriever.Config(
                model=model,
                tokenizer=tokenizer,
            )
        )
        retriever.build_index("dense_test")
        loaded_retriever = DenseRetriever(
            dictionary=processed_dictionary,
            config=DenseRetriever.Config(
                model=model,
                tokenizer=tokenizer,
            )
        )
        loaded_retriever.build_index("dense_test")
        assert retriever.dictionary and loaded_retriever.dictionary
        assert retriever.meta_ids_to_keys == loaded_retriever.meta_ids_to_keys
        assert retriever.dictionary.id_to_index == loaded_retriever.dictionary.id_to_index

    @pytest.mark.parametrize("model", MODELS)
    def test_len(self, model: DualBERTModel) -> None:
        model.resize_token_embeddings(len(tokenizer))
        processed_dictionary = dictionary_preprocess(dictionary)
        retriever = DenseRetriever(
            dictionary=processed_dictionary,
            config=DenseRetriever.Config(
                model=model,
                tokenizer=tokenizer
            )
        )
        retriever.build_index("dense_test")
        assert len(retriever) == len(dictionary)

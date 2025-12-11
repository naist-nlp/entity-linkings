from importlib.resources import files

import pytest
from torch.utils.data import DataLoader

import assets as test_data
from entity_linkings import get_retrievers, load_dataset, load_dictionary
from entity_linkings.models import (
    BLINK,
    EXTEND,
    FEVRY,
    FUSIONED,
    SpanEntityRetrievalForDualEncoder,
)

from .collator import (
    CollatorForCrossEncoder,
    CollatorForExtend,
    CollatorForFusioned,
    CollatorForReranking,
    CollatorForRetrieval,
)

dictionary_path = str(files(test_data).joinpath("dictionary_toy.jsonl"))
dictionary = load_dictionary(dictionary_path)
dataset_path = str(files(test_data).joinpath("dataset_toy.jsonl"))
dataset = load_dataset("json", dataset_path, split="train", cache_dir='.cache_test')
retriever_cls = get_retrievers("bm25")
retriever = retriever_cls(dictionary=dictionary)


@pytest.mark.parametrize("model_cls", [SpanEntityRetrievalForDualEncoder])
def test_CollatorForRetrieval(model_cls: type) -> None:
    model = model_cls(dictionary=dictionary)
    candidates_ids = retriever.retrieve_candidates(dataset, top_k=3, negative=True)
    processed_dataset = model.data_preprocess(dataset)
    processed_dataset = processed_dataset.add_column("candidates", candidates_ids)
    collator = CollatorForRetrieval(model.tokenizer, dictionary=dictionary, num_hard_negatives=3)
    dataloader = DataLoader(processed_dataset, batch_size=2, collate_fn=collator)

    for batch in dataloader:
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "token_type_ids" in batch
        assert "candidates_input_ids" in batch
        assert "candidates_attention_mask" in batch
        assert "candidates_token_type_ids" in batch
        assert "labels" in batch
        assert batch["input_ids"].size(0) == batch["attention_mask"].size(0) == 2
        assert batch["candidates_input_ids"].size(0) == batch["candidates_attention_mask"].size(0) == 2


@pytest.mark.parametrize("model_cls", [FEVRY])
@pytest.mark.parametrize("train", [True, False])
def test_CollatorForReranking(model_cls: type, train: bool) -> None:
    model = model_cls(retriever=retriever)
    candidates_ids = retriever.retrieve_candidates(dataset, top_k=3, only_negative=True)
    processed_dataset = model.data_preprocess(dataset)
    processed_dataset = processed_dataset.add_column("candidates", candidates_ids)
    collator = CollatorForReranking(model.tokenizer, dictionary=dictionary, train=train)
    dataloader = DataLoader(processed_dataset, batch_size=2, collate_fn=collator)

    for batch in dataloader:
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch
        assert batch["input_ids"].size(0) == batch["attention_mask"].size(0) == 2
        if "start_positions" in batch:
            assert batch["start_positions"].size() == (2, )
            assert batch["end_positions"].size() == (2, )
        if train:
            assert batch["candidates_ids"].size() == (2, 4)
        else:
            assert batch["candidates_ids"].size() == (2, 3)
        assert batch["labels"].size() == (2, )


@pytest.mark.parametrize("model_cls", [BLINK])
@pytest.mark.parametrize("train", [True, False])
def test_CollatorForCrossEncoder(model_cls: type, train: bool) -> None:
    model = model_cls(retriever=retriever)
    candidates_ids = retriever.retrieve_candidates(dataset, top_k=3, only_negative=True)
    processed_dataset = model.data_preprocess(dataset)
    processed_dataset = processed_dataset.add_column("candidates", candidates_ids)
    collator = CollatorForCrossEncoder(model.tokenizer, dictionary=dictionary, train=train)
    dataloader = DataLoader(processed_dataset, batch_size=2, collate_fn=collator)

    for batch in dataloader:
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch
        if train:
            assert batch["input_ids"].size(0) == batch["attention_mask"].size(0) == 2
            assert batch["input_ids"].size(1) == batch["attention_mask"].size(1) == 4
            assert batch["labels"].size() == (2, 4)
        else:
            assert batch["input_ids"].size(0) == batch["attention_mask"].size(0) == 2
            assert batch["input_ids"].size(1) == batch["attention_mask"].size(1) == 3
            assert batch["labels"].size() == (2, 3)


@pytest.mark.parametrize("model_cls", [EXTEND])
@pytest.mark.parametrize("train", [True, False])
def test_CollatorForExtend(model_cls: type, train: bool) -> None:
    model = model_cls(retriever=retriever)
    candidates_ids = retriever.retrieve_candidates(dataset, top_k=3, only_negative=True)
    filtered_dataset = model.data_filter(dataset)
    processed_dataset = filtered_dataset.add_column("candidates", candidates_ids)
    processed_dataset = model.data_preprocess(processed_dataset, train=train)

    collator = CollatorForExtend(model.tokenizer)
    dataloader = DataLoader(processed_dataset, batch_size=4, collate_fn=collator)

    for batch in dataloader:
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "candidates_offsets" in batch
        assert batch["input_ids"].size(0) == batch["attention_mask"].size(0) == 4
        assert batch["input_ids"].size(1) == batch["attention_mask"].size(1)
        assert batch["candidates_offsets"].size() == (4, 3, 2)
        if train:
            assert "labels" in batch
            assert batch["labels"].size() == (2, 4)
        else:
            assert "labels" not in batch


@pytest.mark.parametrize("model_cls", [FUSIONED])
@pytest.mark.parametrize("train", [True, False])
def test_CollatorForFusioned(model_cls: type, train: bool) -> None:
    model = model_cls(retriever=retriever)
    candidates_ids = retriever.retrieve_candidates(dataset, top_k=3, only_negative=True)
    processed_dataset = model.data_preprocess(dataset)
    processed_dataset = processed_dataset.add_column("candidates", candidates_ids)
    collator = CollatorForFusioned(model.tokenizer, dictionary=dictionary, train=train)
    dataloader = DataLoader(processed_dataset, batch_size=2, collate_fn=collator)

    for batch in dataloader:
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch
        assert batch["input_ids"].size(0) == batch["attention_mask"].size(0) == 2
        assert batch["input_ids"].size(1) == batch["attention_mask"].size(1) == 3
        assert batch["labels"].size(0) == 2

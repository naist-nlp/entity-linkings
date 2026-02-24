from importlib.resources import files

import pytest
from datasets import load_dataset
from torch.utils.data import DataLoader

import assets as test_data
from entity_linkings import get_retrievers, load_dictionary

from .collator import CollatorForFusioned
from .fusioned import FUSIONED

dictionary_path = str(files(test_data).joinpath("dictionary_toy.jsonl"))
dictionary = load_dictionary(dictionary_path)
dataset_path = str(files(test_data).joinpath("dataset_toy.jsonl"))
dataset = load_dataset("json", data_files={"test": dataset_path})['test']
retriever_cls = get_retrievers("bm25")
retriever = retriever_cls(dictionary=dictionary)


@pytest.mark.parametrize("train", [True, False])
def test_CollatorForFusioned(train: bool) -> None:
    model = FUSIONED(retriever=retriever)
    candidates_ids = retriever.retrieve_candidates(dataset, top_k=3, only_negative=True if train else False)
    processed_dataset = model.preprocessor.dataset_preprocess(dataset, candidates_ids)
    collator = CollatorForFusioned(model.tokenizer, dictionary=dictionary, train=train)
    dataloader = DataLoader(processed_dataset, batch_size=2, collate_fn=collator)

    for batch in dataloader:
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch
        assert batch["input_ids"].size(0) == batch["attention_mask"].size(0) == 2
        assert batch["input_ids"].size(1) == batch["attention_mask"].size(1) == 3
        assert batch["labels"].size(0) == 2

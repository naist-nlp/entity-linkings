from importlib.resources import files

from datasets import load_dataset
from transformers import AutoTokenizer

import assets as test_data
from entity_linkings import get_retrievers, load_dictionary

from .collator import CollatorBase

dictionary_path = str(files(test_data).joinpath("dictionary_toy.jsonl"))
dictionary = load_dictionary(dictionary_path)
dataset_path = str(files(test_data).joinpath("dataset_toy.jsonl"))
dataset = load_dataset("json", data_files={"test": dataset_path})['test']
retriever_cls = get_retrievers("bm25")
retriever = retriever_cls(dictionary=dictionary)


def test_CollatorBase() -> None:
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    collator = CollatorBase(tokenizer=tokenizer)
    assert collator.tokenizer == tokenizer

    encodings = [tokenizer("This is a test."), tokenizer("Another test.")]
    batch = collator(encodings)
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert batch["input_ids"].size(0) == batch["attention_mask"].size(0) == 2
    assert batch["input_ids"].size(1) == batch["attention_mask"].size(1)

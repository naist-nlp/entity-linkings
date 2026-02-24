from importlib.resources import files

import pytest
from datasets import load_dataset
from torch.utils.data import DataLoader

import assets as test_data
from entity_linkings import get_retrievers, load_dictionary

from .collator import CollatorForExtend
from .extend import EXTEND

dictionary_path = str(files(test_data).joinpath("dictionary_toy.jsonl"))
dictionary = load_dictionary(dictionary_path)
dataset_path = str(files(test_data).joinpath("dataset_toy.jsonl"))
dataset = load_dataset("json", data_files={"test": dataset_path})['test']
retriever_cls = get_retrievers("bm25")
retriever = retriever_cls(dictionary=dictionary)


@pytest.mark.parametrize("modify_global_attention", [1, 2])
def test_CollatorForExtend(modify_global_attention: int) -> None:
    model = EXTEND(retriever=retriever)
    candidates_ids = retriever.retrieve_candidates(dataset, top_k=3, only_negative=True)
    processed_dataset = model.preprocessor.dataset_preprocess(dataset, candidates_ids)
    collator = CollatorForExtend(model.tokenizer, modify_global_attention=modify_global_attention)
    dataloader = DataLoader(processed_dataset, batch_size=4, collate_fn=collator)

    for batch in dataloader:
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert batch["input_ids"].size(0) == batch["attention_mask"].size(0) == 4
        assert batch["input_ids"].size(1) == batch["attention_mask"].size(1)
        assert batch["global_attention_mask"].size() == batch["input_ids"].size()
        if "start_positions" in batch and "end_positions" in batch:
            print(batch["start_positions"].size(), batch["end_positions"].size())
            assert batch["start_positions"].size(0) == batch["end_positions"].size(0) == 4
        assert "labels" not in batch

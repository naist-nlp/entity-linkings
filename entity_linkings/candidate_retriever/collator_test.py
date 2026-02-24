from importlib.resources import files

from datasets import load_dataset
from torch.utils.data import DataLoader

import assets as test_data
from entity_linkings import get_retrievers, load_dictionary

from .collator import CollatorForRetrieval

dictionary_path = str(files(test_data).joinpath("dictionary_toy.jsonl"))
dictionary = load_dictionary(dictionary_path)
dataset_path = str(files(test_data).joinpath("dataset_toy.jsonl"))
dataset = load_dataset("json", data_files={"test": dataset_path})['test']
retriever_cls = get_retrievers("dualencoder")
retriever = retriever_cls(dictionary=dictionary)
candidates_ids = retriever.retrieve_candidates(dataset, top_k=3, negative=True)

def test_CollatorForRetrieval() -> None:
    processed_dataset = retriever.preprocessor.dataset_preprocess(dataset, candidates_ids)
    collator = CollatorForRetrieval(retriever.tokenizer, dictionary=dictionary, num_hard_negatives=2)
    dataloader = DataLoader(processed_dataset, batch_size=2, collate_fn=collator)

    for batch in dataloader:
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "token_type_ids" in batch
        assert "candidates_input_ids" in batch
        assert "candidates_attention_mask" in batch
        assert "candidates_token_type_ids" in batch
        assert "labels" in batch
        assert batch["input_ids"].size(0) == batch["attention_mask"].size(0) == batch["labels"].size(0) == 2
        assert batch["candidates_input_ids"].size(0) == batch["candidates_attention_mask"].size(0)
        assert batch["candidates_input_ids"].size(0) >= 2 and batch["candidates_input_ids"].size(0) <= 4
        assert batch['labels'].size() == (2, )

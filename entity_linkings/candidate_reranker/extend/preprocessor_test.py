from importlib.resources import files

import pytest
from datasets import load_dataset
from transformers import AutoTokenizer

import assets as test_data
from entity_linkings import get_retrievers, load_dictionary

from .preprocessor import ExtendPreprocessor, compute_char_to_tokens, process_candidates

MODEL = "allenai/longformer-base-4096"
dictionary_path = str(files(test_data).joinpath("dictionary_toy.jsonl"))
dictionary = load_dictionary(dictionary_path)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.add_special_tokens({'additional_special_tokens': ['[START_ENT]', '[END_ENT]', '[NIL]']})
assert tokenizer.is_fast

preprocessor = ExtendPreprocessor(
    tokenizer=tokenizer,
    dictionary=dictionary,
    ent_start_token="[START_ENT]",
    ent_end_token="[END_ENT]",
    entity_token="[ENT]",
    max_context_length=512,
    max_candidate_length=128,
)
retriever_cls = get_retrievers("bm25")
retriever = retriever_cls(dictionary=dictionary)
sentence = "Apple is looking at buying U.K. startup for $1 billion"
spans = [(0,5), (27,30), (44,52)]
candidates = retriever.predict(sentence, spans=spans, top_k=3)
candidate_ids = [[cand.id for cand in candidates_i] for candidates_i in candidates]

@pytest.mark.parametrize("labels", [["000013"], None])
def test_process_context(labels: list[str]| None) -> None:
    for (b, e), candidates_i in zip(spans, candidate_ids):
        encodings = preprocessor.process_context(sentence, b, e, candidates_i, labels)
        assert "input_ids" in encodings
        assert "attention_mask" in encodings
        assert "offset_mapping" in encodings
        assert "candidates_offsets" in encodings
        if labels is not None:
            assert "labels" in encodings and encodings["labels"]

def test_dataset_preprocess() -> None:
    dataset_path = str(files(test_data).joinpath("dataset_toy.jsonl"))
    dataset = load_dataset("json", data_files={"test": dataset_path})['test']
    candidates = retriever.retrieve_candidates(dataset, top_k=3, only_negative=True)
    processed_dataset = preprocessor.dataset_preprocess(dataset, candidates)
    assert len(processed_dataset) == 8
    for processed in processed_dataset:
        assert "input_ids" in processed
        assert "attention_mask" in processed
        assert "candidates_offsets" in processed
        assert "labels" in processed and processed["labels"]


@pytest.mark.parametrize("gold_titles", [["Meta"], []])
def test_process_candidates(gold_titles: list[str]) -> None:
    candidate_titles = ["Apple", "Meta", "Amazon"]
    gold_titles = ["Meta", "Google"]
    context, answer_starts, answer_ends, candidates_offsets = process_candidates(
        candidate_titles,
        gold_titles,
        separator='*'
    )
    assert context == "Apple * Meta * Amazon * "
    assert answer_starts == [8] if gold_titles else []
    assert answer_ends == [12] if gold_titles else []
    assert candidates_offsets == [(0, 5), (8, 12), (15, 21)]


def test_compute_char_to_tokens() -> None:
    text = "Steve Jobs was found [START_ENT]Apple[END_ENT]."
    candidate_text = "Apple * Meta * Amazon * Google *"

    encodings = tokenizer(
        text,
        candidate_text,
        return_offsets_mapping=True,
    )

    char2token = compute_char_to_tokens(
        candidate_text,
        [p == 1 for p in encodings.sequence_ids()],
        encodings['offset_mapping']
    )
    assert isinstance(char2token, dict)

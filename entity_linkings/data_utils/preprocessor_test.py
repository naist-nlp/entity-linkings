from importlib.resources import files

import pytest
from datasets import load_dataset
from transformers import AutoTokenizer

import assets as test_data
from entity_linkings import get_retrievers, load_dictionary

from .preprocessor import Preprocessor

MODELS = [
    "google-bert/bert-base-uncased",
    "FacebookAI/xlm-roberta-base",
    "microsoft/deberta-v3-base",
    "FacebookAI/roberta-base",
    "answerdotai/ModernBERT-base",
    "google-t5/t5-small",
]

dictionary_path = str(files(test_data).joinpath("dictionary_toy.jsonl"))
dictionary = load_dictionary(dictionary_path)
dataset_path = str(files(test_data).joinpath("dataset_toy.jsonl"))
dataset = load_dataset("json", data_files={"test": dataset_path})['test']

retriever_cls = get_retrievers("bm25")
retriever = retriever_cls(dictionary=dictionary)
candidates_ids = retriever.retrieve_candidates(dataset, top_k=3, only_negative=True)

class TestPreprocessor:
    @pytest.mark.parametrize("model_name", MODELS)
    def test_init(self, model_name: str) -> None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.add_special_tokens({"additional_special_tokens": ["[ENT_START]", "[ENT_END]", "[ENTITY]"]})
        preprocessor = Preprocessor(
            tokenizer=tokenizer,
            ent_start_token="[ENT_START]",
            ent_end_token="[ENT_END]",
            entity_token="[ENTITY]",
            max_context_length=512,
            max_candidate_length=32,
            context_window_chars=100,
        )
        assert preprocessor.ent_start_token == "[ENT_START]"
        assert preprocessor.ent_end_token == "[ENT_END]"
        assert preprocessor.entity_token == "[ENTITY]"
        assert len(preprocessor.prefix_ids) == 0 if model_name.startswith("google-t5") else 1
        assert len(preprocessor.suffix_ids) == 1
        assert preprocessor.max_context_length == 512 - len(preprocessor.prefix_ids) - len(preprocessor.suffix_ids)
        assert preprocessor.offset_correction == len(preprocessor.prefix_ids)
        assert preprocessor.max_candidate_length == 32
        assert preprocessor.context_window_chars == 100

    def test_dictionary_preprocess(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained(MODELS[0])
        tokenizer.add_special_tokens({"additional_special_tokens": ["[ENT_START]", "[ENT_END]", "[ENTITY]"]})
        preprocessor = Preprocessor(tokenizer=tokenizer, ent_start_token="[ENT_START]", ent_end_token="[ENT_END]", entity_token="[ENTITY]", max_candidate_length=32)
        processed_dictionary = preprocessor.dictionary_preprocess(dictionary)
        for entry in processed_dictionary:
            assert "encoding" in entry
            assert isinstance(entry["encoding"], list)
            assert len(entry["encoding"]) <= 32

    def test_process_context(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained(MODELS[0])
        tokenizer.add_special_tokens({"additional_special_tokens": ["[ENT_START]", "[ENT_END]", "[ENTITY]"]})
        preprocessor = Preprocessor(tokenizer=tokenizer, ent_start_token="[ENT_START]", ent_end_token="[ENT_END]", entity_token="[ENTITY]", max_context_length=128, context_window_chars=200)
        text = dataset['text'][0]
        entity = dataset['entities'][0][0]
        encodings = preprocessor.process_context(text, entity['start'], entity['end'])
        assert "input_ids" in encodings
        assert "attention_mask" in encodings
        start_pos = encodings['input_ids'].index(preprocessor.start_marker_id)
        end_pos = encodings['input_ids'].index(preprocessor.end_marker_id)
        mention_ids = encodings['input_ids'][start_pos: end_pos + 1]
        mention_text = text[entity['start']: entity['end']].lower()
        decoded_mention = tokenizer.decode(mention_ids, skip_special_tokens=False).replace(" ", "")
        expected_mention = f"{preprocessor.ent_start_token}{mention_text}{preprocessor.ent_end_token}".replace(" ", "")
        assert decoded_mention == expected_mention
        assert len(encodings["input_ids"]) == len(encodings["attention_mask"])
        assert len(encodings["input_ids"]) <= preprocessor.max_context_length

    def test_data_flatten(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained(MODELS[0])
        tokenizer.add_special_tokens({"additional_special_tokens": ["[ENT_START]", "[ENT_END]", "[ENTITY]"]})
        preprocessor = Preprocessor(tokenizer=tokenizer, ent_start_token="[ENT_START]", ent_end_token="[ENT_END]", entity_token="[ENTITY]", max_candidate_length=32)
        flattened_dataset = preprocessor.data_flatten(dataset)
        assert len(flattened_dataset) == 8
        assert len(dataset) == 8
        assert len(set([d['id'] for d in flattened_dataset])) == 7

    def test_dataset_preprocess(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained(MODELS[0])
        tokenizer.add_special_tokens({"additional_special_tokens": ["[ENT_START]", "[ENT_END]", "[ENTITY]"]})
        preprocessor = Preprocessor(
            tokenizer=tokenizer,
            ent_start_token="[ENT_START]",
            ent_end_token="[ENT_END]",
            entity_token="[ENTITY]",
            max_context_length=128,
            max_candidate_length=32,
            context_window_chars=200,
        )
        processed_dataset = preprocessor.dataset_preprocess(dataset, candidates_ids)
        assert len(processed_dataset) == 8
        for processed in processed_dataset:
            assert "input_ids" in processed
            assert "attention_mask" in processed
            assert "start_positions" in processed
            assert "end_positions" in processed
            assert "candidates" in processed
            assert "labels" in processed

        processed_dataset = preprocessor.dataset_preprocess(dataset)
        assert len(processed_dataset) == 8
        for processed in processed_dataset:
            assert "input_ids" in processed
            assert "attention_mask" in processed
            assert "start_positions" in processed
            assert "end_positions" in processed
            assert "candidates" not in processed
            assert "labels" in processed

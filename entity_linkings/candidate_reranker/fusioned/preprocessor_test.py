from importlib.resources import files

from datasets import load_dataset
from transformers import AutoTokenizer

import assets as test_data
from entity_linkings import get_retrievers, load_dictionary

from .preprocessor import FusionedPreprocessor

MODELS = ["google/flan-t5-base"]
dictionary_path = str(files(test_data).joinpath("dictionary_toy.jsonl"))
dictionary = load_dictionary(dictionary_path)
dataset_path = str(files(test_data).joinpath("dataset_toy.jsonl"))
dataset = load_dataset("json", data_files={"test": dataset_path})['test']
retriever_cls = get_retrievers("bm25")
retriever = retriever_cls(dictionary=dictionary)
candidates_ids = retriever.retrieve_candidates(dataset, top_k=3, only_negative=True)


class TestFusionedPreprocessor:
    def test_init(self) -> None:
        model_name = MODELS[0]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        preprocessor = FusionedPreprocessor(
            tokenizer=tokenizer,
            ent_start_token="<extra_id_6>",
            ent_end_token="<extra_id_7>",
            title_token="<extra_id_2>",
            entity_token="<extra_id_4>",
        )
        assert preprocessor.title_token == "<extra_id_2>"
        assert preprocessor.ent_start_token == "<extra_id_6>"
        assert preprocessor.ent_end_token == "<extra_id_7>"
        assert preprocessor.entity_token == "<extra_id_4>"
        assert len(preprocessor.prefix_ids) == 0
        assert len(preprocessor.suffix_ids) == 1
        assert preprocessor.max_context_length == 249 # 250 - len(preprocessor.prefix_ids) - len(preprocessor.suffix_ids)
        assert preprocessor.offset_correction == len(preprocessor.prefix_ids)
        assert preprocessor.max_candidate_length == 100
        assert preprocessor.context_window_chars == 1000

    def test_dictionary_preprocess(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained(MODELS[0])
        preprocessor = FusionedPreprocessor(
            tokenizer=tokenizer,
            ent_start_token="<extra_id_6>",
            ent_end_token="<extra_id_7>",
            title_token="<extra_id_2>",
            entity_token="<extra_id_4>",
        )
        processed_dictionary = preprocessor.dictionary_preprocess(dictionary)
        for entry in processed_dictionary:
            assert "encoding" in entry
            assert isinstance(entry["encoding"], list)
            tokens = tokenizer.convert_ids_to_tokens(entry["encoding"])
            assert tokens[0] == "<extra_id_2>"
            assert tokenizer.decode(entry["encoding"]) == f"<extra_id_2> {entry['name']}<extra_id_4> {entry['description']}"
            assert len(entry["encoding"]) <= 32


    def test_process_context(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained(MODELS[0])
        preprocessor = FusionedPreprocessor(
            tokenizer=tokenizer,
            ent_start_token="<extra_id_6>",
            ent_end_token="<extra_id_7>",
            title_token="<extra_id_2>",
            entity_token="<extra_id_4>",
        )
        text = dataset['text'][0]
        entity = dataset['entities'][0][0]
        encodings = preprocessor.process_context(text, entity['start'], entity['end'])
        assert "input_ids" in encodings
        assert "attention_mask" in encodings
        assert "start_positions" not in encodings
        assert "end_positions" not in encodings
        start_pos = encodings['input_ids'].index(preprocessor.start_marker_id)
        end_pos = encodings['input_ids'].index(preprocessor.end_marker_id)
        mention_ids = encodings['input_ids'][start_pos: end_pos + 1]
        mention_text = text[entity['start']: entity['end']]
        decoded_mention = tokenizer.decode(mention_ids, skip_special_tokens=False).replace(" ", "")
        expected_mention = f"{preprocessor.ent_start_token}{mention_text}{preprocessor.ent_end_token}".replace(" ", "")
        assert decoded_mention == expected_mention
        assert len(encodings["input_ids"]) == len(encodings["attention_mask"])
        assert len(encodings["input_ids"]) <= preprocessor.max_context_length


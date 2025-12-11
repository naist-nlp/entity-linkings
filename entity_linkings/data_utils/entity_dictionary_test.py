from importlib.resources import files

from datasets import Column, load_dataset

import assets as test_data
from entity_linkings import load_dictionary

from .entity_dictionary import EntityDictionary

dictionary_path = str(files(test_data).joinpath("dictionary_toy.jsonl"))


class TestEntityDictionary:
    def test__init__(self) -> None:
        dictionary = load_dataset("json", data_files=dictionary_path, split="train")
        dictionary = EntityDictionary(dictionary)
        assert isinstance(dictionary, EntityDictionary)
        assert dictionary.cache_dir is None
        assert dictionary.nil_id == "-1"
        assert dictionary.nil_name == "<NIL>"
        assert dictionary.nil_description == "<NIL> is an entity that does not exist in this dictionary."
        assert dictionary.id_to_index == {"000011": 0, "000012": 1, "000013": 2, "000014": 3, "000015": 4, "-1": 5}
        assert isinstance(dictionary("000012"), dict) and dictionary("000012")["name"] == "Apple"
        assert isinstance(dictionary[0], dict) and dictionary[0]["name"] == "Google"
        assert len(dictionary) == 6

    def test_get_entity_ids(self) -> None:
        dictionary = load_dictionary(dictionary_path)
        ids = dictionary.get_entity_ids()
        assert isinstance(ids, Column)
        assert ids == ["000011", "000012", "000013", "000014", "000015", "-1"]

    def test_get_entity_names(self) -> None:
        dictionary = load_dictionary(dictionary_path)
        names = dictionary.get_entity_names()
        assert isinstance(names, Column)
        assert len(names) == len(dictionary)
        assert names[0] == "Google"
        assert names[1] == "Apple"
        assert names[2] == "Meta"
        assert names[3] == "Amazon"
        assert names[4] == "Microsoft"
        assert names[5] == "[NIL]"

    def test_get_entity_descriptions(self) -> None:
        dictionary = load_dictionary(dictionary_path)
        descriptions = dictionary.get_entity_descriptions()
        assert isinstance(descriptions, Column)
        assert len(descriptions) == len(dictionary)
        assert descriptions[0] == "Google is a global company"
        assert descriptions[1] == "Apple is a global company"
        assert descriptions[2] == "Meta is a global company"
        assert descriptions[3] == "Amazon is a global company"
        assert descriptions[4] == "Microsoft is a global company"
        assert descriptions[5] == "[NIL] is an entity that does not exist in this dictionary."

    def test_convert_description(self) -> None:
        dictionary = load_dictionary(dictionary_path)
        description = dictionary.convert_description("TestEntity", "This is a test entity.")
        assert description == "This is a test entity."
        description = dictionary.convert_description("TestEntity", None)
        assert description == "TestEntity is an entity in this dictionary."

    def test_iter(self) -> None:
        dictionary = load_dictionary(dictionary_path)
        for i, entity in enumerate(dictionary):
            assert isinstance(entity, dict)


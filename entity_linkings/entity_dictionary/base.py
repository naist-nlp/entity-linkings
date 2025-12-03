import abc
import os
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional, TypedDict

import datasets
from datasets import Column, Dataset
from transformers import TrainingArguments

VERSION = datasets.Version("1.0.0")

dictionary_features = datasets.Features(
    {
        "dictionary": datasets.Value("string"),
        "id": datasets.Value("string"),
        "name": datasets.Value("string"),
        "description": datasets.Value("string"),
    }
)

class Entity(TypedDict):
    id: str
    name: str
    description: str
    label_id: int
    encoding: Optional[dict[str, list[int]]]


@dataclass
class EntityDictionaryConfig(datasets.BuilderConfig):
    """BuilderConfig for EntityDictionary."""
    name: Optional[str] = None
    version: Optional[datasets.Version] = None
    description: Optional[str] = None


class EntityDictionaryBase(abc.ABC):

    @dataclass
    class Config:
        nil_id: str = "-1"
        nil_name: str = "<NIL>"
        nil_description: str = "<NIL> is an entity that does not exist in this dictionary."
        default_description: str = """{name} is an entity in this dictionary."""
        cache_dir: Optional[str|os.PathLike] = None

    def __init__(self, dictionary: Dataset, config: Optional[Config] = None) -> None:
        self.config = config if config is not None else self.Config()
        self.entity_dict = dictionary
        self.entity_dict = self.entity_dict.add_item({
            "id": self.config.nil_id,
            "name": self.config.nil_name,
            "description": self.config.nil_description
        })
        self.id_to_index = {id: i for i, id in enumerate(self.entity_dict["id"])}
        self.title_to_index = {title: i for i, title in enumerate(self.entity_dict["name"])}

    def __call__(self, entity_id: str) -> Entity:
        if entity_id in self.id_to_index:
            index = self.id_to_index[entity_id]
            return self.entity_dict[index]
        else:
            nil_index = self.id_to_index[self.config.nil_id]
            return self.entity_dict[nil_index]

    def __iter__(self) -> Iterator[Entity]:
        for value in self.entity_dict:
            yield value

    def __getitem__(self, idx: int) -> Entity:
        return self.entity_dict[idx]

    def __len__(self) -> int:
        return len(self.entity_dict)

    def __repr__(self) -> str:
        return self.entity_dict.__repr__()

    def get_label_id(self, entity_id: str) -> int:
        if entity_id in self.id_to_index:
            return self.id_to_index[entity_id]
        else:
            return self.id_to_index[self.config.nil_id]

    def get_from_title(self, title: str) -> Entity:
        if title in self.title_to_index:
            index = self.title_to_index[title]
            return self.entity_dict[index]
        else:
            nil_index = self.title_to_index[self.config.nil_name]
            return self.entity_dict[nil_index]

    def get_entity_ids(self) -> Column:
        return self.entity_dict["id"]

    def get_entity_names(self) -> Column:
        return self.entity_dict["name"]

    def get_entity_descriptions(self) -> Column:
        return self.entity_dict["description"]

    def convert_description(self, name: str, description: Optional[str] = None) -> str:
        if description:
            return description
        else:
            return self.config.default_description.format(name=name)

    def add_encoding(
            self,
            tokenizer_func: Callable[[str, str], dict[str, list[int]]],
            training_arguments: Optional[TrainingArguments] = None
        ) -> None:
        def preprocess(documents: Dataset) -> dict[str, list[Any]]:
            outputs: dict[str, list[Any]] = {"id": [], "name": [], "description": [], "encoding": []}
            for id, name, description in zip(documents["id"], documents["name"], documents["description"]):
                description = self.convert_description(name, description)
                outputs["id"].append(id)
                outputs["name"].append(name)
                outputs["description"].append(description)
                outputs["encoding"].append(tokenizer_func(name, description))
            return outputs

        if training_arguments:
            with training_arguments.main_process_first(desc="dataset map pre-processing"):
                column_names = next(iter(self.entity_dict)).column_names
                dictionary = self.entity_dict.map(preprocess, batched=True, remove_columns=column_names)
        else:
            column_names = self.entity_dict.column_names
            dictionary = self.entity_dict.map(preprocess, batched=True, remove_columns=column_names)

        self.entity_dict = dictionary

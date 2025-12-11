import os
from typing import Optional, Union

import datasets
from datasets import Dataset, DatasetDict

from .data_utils import EntityDictionary
from .dataset import DATASET_ID2CLS
from .models import (
    ED_ID2CLS,
    EL_ID2CLS,
    RETRIEVER_ID2CLS,
    EntityRetrieverBase,
    PipelineBase,
)


def get_retriever_ids() -> list[str]:
    '''Generate a list of ids with the class name in lower case.
    '''
    ids = list(RETRIEVER_ID2CLS.keys())
    return ids


def get_el_ids() -> list[str]:
    '''Generate a list of ids with the class name in lower case.
    '''
    ids = list(EL_ID2CLS.keys())
    return ids


def get_ed_ids() -> list[str]:
    '''Generate a list of ids with the class name in lower case.
    '''
    ids = list(ED_ID2CLS.keys())
    return ids


def get_model_ids() -> list[str]:
    '''Generate a list of ids with the class name in lower case.
    '''
    ids = list(RETRIEVER_ID2CLS.keys()) + list(EL_ID2CLS.keys()) + list(ED_ID2CLS.keys())
    return ids


def load_dataset(
        name: str = "json",
        data_files: Optional[Union[str, dict[str, str]]] = None,
        split: Optional[str] = None,
        cache_dir: Optional[str] = None
    ) -> Union[DatasetDict, Dataset]:
    '''Generate a dataset class with the class name in lower case as the key.
    If the name is not found, use the custom dataset class.
    For custom dataset, data_files must be provided.
    '''
    if name == "json":
        if not data_files:
            raise ValueError("Either name or data_files must be provided.")
        dataset = datasets.load_dataset("json", data_files=data_files, cache_dir=cache_dir)
    elif name in ["zelda", "kilt", "zeshel", "unseen", "tweeki", "reddit-comments", "reddit-posts", "wned-wiki", "wned-cweb"] or name.startswith("naist-nlp/"):
        subset = str(name.split('-')[1]) if '-' in name else None
        name = name.split('-')[0]
        if not name.startswith("naist-nlp/"):
            name = f"naist-nlp/{name}"
        if subset:
            dataset = datasets.load_dataset(name, subset, cache_dir=cache_dir)
        else:
            dataset = datasets.load_dataset(name, cache_dir=cache_dir)
    else:
        subset = str(name.split('-')[1]) if '-' in name else None
        name = name.split('-')[0]
        if name not in DATASET_ID2CLS:
            raise ValueError(f"The id should be one of {list(DATASET_ID2CLS.keys())}.")
        dataset_cls = DATASET_ID2CLS[name]
        if subset:
            dataset_cls(config_name=subset, cache_dir=cache_dir).download_and_prepare()
            dataset = dataset_cls(config_name=subset, cache_dir=cache_dir).as_dataset()
        else:
            dataset_cls(cache_dir=cache_dir).download_and_prepare()
            dataset = dataset_cls(cache_dir=cache_dir).as_dataset()

    if split is not None:
        return dataset[split]
    return dataset


def load_dictionary(
        dictionary_name_or_path: str,
        nil_id: str = "-1",
        nil_name: str = "[NIL]",
        nil_description: str = "[NIL] is an entity that does not exist in this dictionary.",
        default_description: str = """{name} is an entity in this dictionary.""",
        cache_dir: Optional[str|os.PathLike] = None,
    ) -> EntityDictionary:
    '''Generate a dictionary of ids and classes with the class name in lower case as the key.
    '''
    if os.path.isfile(dictionary_name_or_path):
        dictionary = datasets.load_dataset("json", data_files=dictionary_name_or_path, cache_dir=cache_dir, split="train")
    else:
        if not dictionary_name_or_path.startswith("naist-nlp/"):
            dictionary_name_or_path = f"naist-nlp/{dictionary_name_or_path}"
        dictionary = datasets.load_dataset(dictionary_name_or_path, 'dictionary', split='kb', cache_dir=cache_dir)

    return EntityDictionary(
        dictionary=dictionary,
        nil_id=nil_id,
        nil_name=nil_name,
        nil_description=nil_description,
        default_description=default_description,
        cache_dir=cache_dir,
    )


def get_ed_models(name: str) -> type[PipelineBase]:
    '''Generate a dictionary of ids and classes with the class name in lower case as the key.
    '''
    if name not in get_ed_ids():
        raise ValueError(f"The id should be one of {get_el_ids()}.")
    return ED_ID2CLS[name]


def get_el_models(name: str) -> type[PipelineBase]:
    '''Generate a dictionary of ids and classes with the class name in lower case as the key.
    '''
    if name not in get_el_ids():
        raise ValueError(f"The id should be one of {get_el_ids()}.")
    return EL_ID2CLS[name]


def get_retrievers(name: str) -> type[EntityRetrieverBase]:
    '''Generate a retriever model class.
    If without_span is True, use SentenceRetrieval class.
    Otherwise, use Retrieval class.
    '''
    if name not in get_retriever_ids():
        raise ValueError(f"The id should be one of {get_retriever_ids()}.")
    return RETRIEVER_ID2CLS[name]


def get_models(name: str) -> type[PipelineBase]:
    '''Generate a dictionary of ids and classes with the class name in lower case as the key.
    '''
    if name in get_el_ids():
        return EL_ID2CLS[name]
    elif name in get_ed_ids():
        return ED_ID2CLS[name]
    else:
        raise ValueError(f"The id should be one of {get_retriever_ids() + get_el_ids() + get_ed_ids()}.")

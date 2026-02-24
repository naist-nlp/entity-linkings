import os
from typing import Optional

import datasets

from .candidate_reranker import RERANKER_ID2CLS, RerankerBase
from .candidate_retriever import RETRIEVER_ID2CLS, RetrieverBase
from .data_utils import EntityDictionary
from .pipeline import ELPipeline


def get_retriever_ids() -> list[str]:
    '''Generate a list of ids with the class name in lower case.
    '''
    ids = list(RETRIEVER_ID2CLS.keys())
    return ids


def get_reranker_ids() -> list[str]:
    '''Generate a list of ids with the class name in lower case.
    '''
    ids = list(RERANKER_ID2CLS.keys())
    return ids


def get_model_ids() -> list[str]:
    '''Generate a list of ids with the class name in lower case.
    '''
    ids = list(RETRIEVER_ID2CLS.keys()) + list(RERANKER_ID2CLS.keys())
    return ids


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


def get_retrievers(name: str) -> type[RetrieverBase]:
    '''Generate a retriever model class.
    If without_span is True, use SentenceRetrieval class.
    Otherwise, use Retrieval class.
    '''
    if name not in get_retriever_ids():
        raise ValueError(f"The id should be one of {get_retriever_ids()}.")
    return RETRIEVER_ID2CLS[name]


def get_rerankers(name: str) -> type[RerankerBase]:
    '''Generate a dictionary of ids and classes with the class name in lower case as the key.
    '''
    if name not in get_reranker_ids():
        raise ValueError(f"The id should be one of {get_reranker_ids()}.")
    return RERANKER_ID2CLS[name]


__all__ = [
    "get_retriever_ids",
    "get_reranker_ids",
    "get_model_ids",
    "load_dictionary",
    "get_retrievers",
    "get_rerankers",
    "EntityDictionary",
    "RetrieverBase",
    "RerankerBase",
    "ELPipeline",
]

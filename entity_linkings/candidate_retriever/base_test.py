from importlib.resources import files

import assets as test_data
from entity_linkings import load_dictionary

from .base import IndexerBase, RetrieverBase

dictionary_path = str(files(test_data).joinpath("dictionary_toy.jsonl"))
dictionary = load_dictionary(dictionary_path)


def test_IndexerBase() -> None:
    indexer = IndexerBase(dictionary=dictionary)
    assert isinstance(indexer, IndexerBase)
    assert indexer.dictionary == dictionary

def test_RetrieverBase() -> None:
    linker = RetrieverBase(dictionary=dictionary)
    assert isinstance(linker, RetrieverBase)
    assert hasattr(linker, 'config')
    assert isinstance(linker.config, RetrieverBase.Config)

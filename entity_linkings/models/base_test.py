from importlib.resources import files

import assets as test_data
from entity_linkings import load_dictionary

from .base import EntityRetrieverBase, IndexerBase, PipelineBase

MODELS = ["google-bert/bert-base-uncased"]
dictionary_path = str(files(test_data).joinpath("dictionary_toy.jsonl"))
dictionary = load_dictionary(dictionary_path)


def test_IndexerBase() -> None:
    indexer = IndexerBase(dictionary=dictionary)
    assert isinstance(indexer, IndexerBase)
    assert hasattr(indexer, 'config')
    assert isinstance(indexer.config, IndexerBase.Config)


def test_EntityRetrieverBase() -> None:
    linker = EntityRetrieverBase(dictionary=dictionary)
    assert isinstance(linker, EntityRetrieverBase)
    assert hasattr(linker, 'config')
    assert isinstance(linker.config, EntityRetrieverBase.Config)


def test_PipelineBase() -> None:
    retriever = EntityRetrieverBase(dictionary=dictionary)
    linker = PipelineBase(retriever=retriever)
    assert isinstance(linker, PipelineBase)
    assert hasattr(linker, 'config')
    assert isinstance(linker.config, PipelineBase.Config)


from .base import (
    EntityRetrieverBase,
    PipelineBase,
)
from .blink import BLINK
from .bm25 import BM25
from .e5bm25 import E5BM25
from .fevry import FEVRY
from .span_retrieval import (
    SpanEntityRetrievalForDualEncoder,
    SpanEntityRetrievalForTextEmbedding,
)
from .zeldacl import ZELDACL

MODEL_BASE_CLS = [
    EntityRetrieverBase,
    PipelineBase
]

RETRIEVER_CLS: list[type[EntityRetrieverBase]] = [
    BM25,
    ZELDACL,
    SpanEntityRetrievalForDualEncoder,
    SpanEntityRetrievalForTextEmbedding,
    E5BM25,
]

ED_CLS: list[type[PipelineBase]] = [
    FEVRY,
    BLINK,
]

EL_CLS: list[type[PipelineBase]] = []

__all__ = [c.__name__ for c in MODEL_BASE_CLS + EL_CLS + ED_CLS]

RETRIEVER_ID2CLS = {
    c.__name__.lower(): c for c in RETRIEVER_CLS
}

ED_ID2CLS = {
    c.__name__.lower(): c for c in ED_CLS
}

EL_ID2CLS = {
    c.__name__.lower(): c for c in EL_CLS
}

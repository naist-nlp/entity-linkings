from .base import RetrieverBase
from .bm25 import BM25
from .dualencoder import DUALENCODER
from .e5bm25 import E5BM25
from .prior import PRIOR
from .textembedding import TEXTEMBEDDING

RETRIEVER_CLS: list[type[RetrieverBase]] = [
    BM25,
    PRIOR,
    TEXTEMBEDDING,
    E5BM25,
    DUALENCODER,
]

RETRIEVER_ID2CLS = {
    c.__name__.lower(): c for c in RETRIEVER_CLS
}

__all__ = [
    "RetrieverBase",
    "RETRIEVER_ID2CLS",
]

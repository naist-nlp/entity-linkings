from .base import RerankerBase
from .chatel import CHATEL
from .crossencoder import CROSSENCODER
from .extend import EXTEND
from .fevry import FEVRY
from .fusioned import FUSIONED

RERANKER_CLS: list[type[RerankerBase]] = [
    CROSSENCODER,
    CHATEL,
    FEVRY,
    EXTEND,
    FUSIONED,
]

RERANKER_ID2CLS = {
    c.__name__.lower(): c for c in RERANKER_CLS
}

__all__ = [
    "RerankerBase",
    "RERANKER_ID2CLS",
]

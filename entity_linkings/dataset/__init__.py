from datasets import GeneratorBasedBuilder

from .ace2004 import ACE2004
from .aquaint import AQUAINT
from .derczynski import DERCZYNSKI
from .docred import DOCRED
from .kore50 import KORE50
from .msnbc import MSNBC
from .n3 import N3
from .oke import OKE
from .shadowlink import SHADOWLINK

DATASET_CLS = [
    ACE2004,
    MSNBC,
    N3,
    DERCZYNSKI,
    DOCRED,
    AQUAINT,
    KORE50,
    OKE,
    SHADOWLINK
]

__all__ = [c.__name__ for c in DATASET_CLS]

DATASET_ID2CLS: dict[str, GeneratorBasedBuilder] = {
    c.__name__.lower(): c for c in DATASET_CLS
}

from datasets import GeneratorBasedBuilder

from .ace2004 import ACE2004
from .aquaint import AQUAINT
from .derczynski import DERCZYNSKI
from .docred import DOCRED
from .kilt import KILT
from .kore50 import KORE50
from .msnbc import MSNBC
from .n3 import N3
from .oke import OKE
from .reddit import REDDIT
from .shadowlink import SHADOWLINK
from .tweeki import TWEEKI
from .unseen import UNSEEN
from .wned import WNED
from .zelda import ZELDA
from .zeshel import ZESHEL

DATASET_CLS = [
    ZELDA,
    ZESHEL,
    KILT,
    REDDIT,
    ACE2004,
    MSNBC,
    N3,
    TWEEKI,
    DERCZYNSKI,
    WNED,
    DOCRED,
    AQUAINT,
    KORE50,
    UNSEEN,
    OKE,
    SHADOWLINK
]

EL_DATASET_CLS = [
    ACE2004,
    MSNBC,
    N3,
    DERCZYNSKI,
    AQUAINT,
    KORE50,
    OKE,
]


__all__ = [c.__name__ for c in DATASET_CLS]

DATASET_ID2CLS: dict[str, GeneratorBasedBuilder] = {
    c.__name__.lower(): c for c in DATASET_CLS
}

EL_DATASET_ID2CLS: dict[str, GeneratorBasedBuilder] = {
    c.__name__.lower(): c for c in EL_DATASET_CLS
}

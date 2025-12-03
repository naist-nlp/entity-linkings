from datasets import GeneratorBasedBuilder

from .base import EntityDictionaryBase
from .kilt_wiki import KILT_WIKI
from .zelda_wiki import ZELDA_WIKI
from .zeshel_wikia import ZESHEL_WIKIA

DICTIONARY_BASE_CLS = [
    EntityDictionaryBase,
]
DICTIONARY_CLS = [
    KILT_WIKI,
    ZESHEL_WIKIA,
    ZELDA_WIKI,
]

__all__ = [c.__name__ for c in DICTIONARY_BASE_CLS + DICTIONARY_CLS]

DICTIONARY_ID2CLS: dict[str, GeneratorBasedBuilder] = {
    c.__name__.lower(): c for c in DICTIONARY_CLS
}

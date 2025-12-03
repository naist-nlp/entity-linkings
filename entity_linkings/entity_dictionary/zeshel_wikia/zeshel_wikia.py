import itertools
import os
from typing import Any, Iterator

import datasets
from datasets import (
    DatasetInfo,
    DownloadManager,
    GeneratorBasedBuilder,
    NamedSplit,
    SplitGenerator,
)

from entity_linkings.dataset.utils import read_jsonl
from entity_linkings.entity_dictionary.base import (
    VERSION,
    EntityDictionaryConfig,
    dictionary_features,
)

_LOCAL = True
_LANGUAGES = ["English'"]
_CITATION = """\
@inproceedings{logeswaran-etal-2019-zero,
    title = "Zero-Shot Entity Linking by Reading Entity Descriptions",
    author = "Logeswaran, Lajanugen and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina and Devlin, Jacob and Lee, Honglak",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    year = "2019"
}
"""

_DATASET_NAME = "zeshel"
_DISPLAY_NAME = "ZESHEL"

_DESCRIPTION = """\
This dataset contains ZESHEL corpus, a zero-shot entity linking benchmark.
"""

_HOMEPAGE = "https://github.com/lajanugen/zeshel"
_URL = "entity_linkings/dataset/zeshel/zeshel.tar.bz2"
_LICENSE = "CC-BY-SA"


logger = datasets.utils.logging.get_logger(__name__)


DOMAIN_SPLITS = {
    "train": [
        "american_football",
        "doctor_who",
        "fallout",
        "final_fantasy",
        "military",
        "pro_wrestling",
        "starwars",
        "world_of_warcraft",
    ],
    "dev": [
        "coronation_street",
        "muppets",
        "ice_hockey",
        "elder_scrolls",
    ],
    "test": [
        "forgotten_realms",
        "lego",
        "star_trek",
        "yugioh"
    ]
}


class ZESHEL_WIKIA(GeneratorBasedBuilder):
    """
    ZESHEL dataset
    """

    BUILDER_CONFIGS = [
        EntityDictionaryConfig(
            name=_DATASET_NAME,
            version=VERSION,
            description=_DESCRIPTION
        ),
    ]

    def _info(self) -> DatasetInfo:
        return DatasetInfo(
            description=_DESCRIPTION,
            features=dictionary_features,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE,
        )

    def _split_generators(self, dl_manager: DownloadManager) -> list[SplitGenerator]:
        """Returns SplitGenerators."""
        return [
            SplitGenerator(
                name=NamedSplit("dictionary"),
                gen_kwargs={"data_dir": os.path.join(dl_manager.download_and_extract(_URL), 'zeshel')},
            ),
        ]

    @staticmethod
    def get_documents(data_dir: str, split: str) -> dict[str, Any]:
        documents = {}
        for domain in DOMAIN_SPLITS[split]:
            file_path = os.path.join(data_dir, f"documents/{domain}.json")
            for line in read_jsonl(file_path):
                documents[line["document_id"]] = {"title": line["title"], "text": line["text"], "domain": domain}
        return documents

    def _generate_examples(self, data_dir: str) -> Iterator[tuple[str, dict[str, Any]]]:
        """"""
        logger.info(f"Generating examples from {data_dir}")
        documents = {split: self.get_documents(data_dir, split) for split in ["train", "dev", "test"]}

        uid = map(str, itertools.count(start=0, step=1))
        for split in documents:
            for doc_id, doc in documents[split].items():
                example = {
                    "dictionary": _DATASET_NAME,
                    "id": doc_id,
                    "name": doc["title"],
                    "description": doc["text"]
                }
                yield next(uid), example

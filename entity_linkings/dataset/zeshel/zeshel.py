import itertools
import os
from typing import Any, Iterator

import datasets
from datasets import (
    DatasetInfo,
    DownloadManager,
    GeneratorBasedBuilder,
    NamedSplit,
    Split,
    SplitGenerator,
)

from ..entity_linkikngs_hub import VERSION, EntityLinkingDatasetConfig, el_features
from ..utils import read_jsonl

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
_LICENCE = "CC-BY-SA"


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


class ZESHEL(GeneratorBasedBuilder):
    """
    ZESHEL dataset
    """

    BUILDER_CONFIGS = [
        EntityLinkingDatasetConfig(
            name=_DATASET_NAME,
            version=VERSION,
            description=_DESCRIPTION,
            subset_id=_DATASET_NAME,
        ),
    ]

    def _info(self) -> DatasetInfo:
        return DatasetInfo(
            description=_DESCRIPTION,
            features=el_features,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: DownloadManager) -> list[SplitGenerator]:
        """Returns SplitGenerators."""
        data_dir = os.path.join(dl_manager.download_and_extract(_URL), "zeshel")
        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={"data_dir": data_dir, "split": "train"},
            ),
            SplitGenerator(
                name=NamedSplit("heldout_train_seen"),
                gen_kwargs={"data_dir": data_dir, "split": "heldout_train_seen"},
            ),
            SplitGenerator(
                name=NamedSplit("heldout_train_unseen"),
                gen_kwargs={"data_dir": data_dir, "split": "heldout_train_unseen"},
            ),
            SplitGenerator(
                name=Split.VALIDATION,
                gen_kwargs={"data_dir": data_dir, "split": "dev"},
            ),
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={"data_dir": data_dir, "split": "test"},
            ),
        ]

    @staticmethod
    def get_documents(input_dir: str, split: str) -> dict[str, Any]:
        documents = {}
        for domain in DOMAIN_SPLITS[split]:
            file_path = os.path.join(input_dir, f"documents/{domain}.json")
            for line in read_jsonl(file_path):
                documents[line["document_id"]] = {"title": line["title"], "text": line["text"], "domain": domain}
        return documents

    @staticmethod
    def get_mention_dictionary(input_dir: str, split: str) -> dict[str, Any]:
        file_path = os.path.join(input_dir, f"mentions/{split}.json")
        mention_dict: dict[str, Any] = {}
        for line in read_jsonl(file_path):
            context_document_id = line["context_document_id"]
            if context_document_id not in mention_dict:
                mention_dict[context_document_id] = []
            mention_dict[context_document_id].append(line)
        return mention_dict

    def _generate_examples(self, data_dir: str, split: str) -> Iterator[tuple[str, dict[str, Any]]]:
        """"""
        logger.info(f"Generating examples for split: {split} from {data_dir}")
        documents = self.get_documents(data_dir, "train" if split in ["train", "heldout_train_seen", "heldout_train_unseen"] else split)
        mention_dict = self.get_mention_dictionary(data_dir, "val" if split == "dev" else split)

        uid = map(str, itertools.count(start=0, step=1))
        for doc_id, doc in documents.items():
            example = {"id": doc_id, "dataset": doc["domain"], "text": doc["text"], "entities": []}
            if doc_id not in mention_dict:
                yield next(uid), example
                continue
            mentions = mention_dict[doc_id]
            for mention in mentions:
                start, end, label = mention["start_index"], mention["end_index"], mention["label_document_id"]
                assert ' '.join(doc["text"].split()[start:end+1]) == mention["text"]

                char_start = len(' '.join(doc["text"].split()[:start])) + 1 if start > 0 else 0
                char_end = len(' '.join(doc["text"].split()[:end+1]))
                assert doc["text"][char_start:char_end] == mention["text"]

                example["entities"].append({
                    "start": char_start,
                    "end": char_end,
                    "label": [label],
                    # "category": mention["category"]
                })
            yield next(uid), example

import itertools
import json
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

from ..base import VERSION, EntityDictionaryConfig, dictionary_features

_LOCAL = False
_LANGUAGES = ["English"]
_CITATION = """\
@inproceedings{milich2023zelda,
    title={{ZELDA}: A Comprehensive Benchmark for Supervised Entity Disambiguation},
    author={Milich, Marcel and Akbik, Alan},
    booktitle={{EACL} 2023,  The 17th Conference of the European Chapter of the Association for Computational Linguistics},
    year={2023}
}
"""

_DATASET_NAME = "zelda"

_DESCRIPTION = """\
This dataset contains ZELDA corpus, a comprehensive benchmark for supervised entity disambiguation.
"""

_HOMEPAGE = "https://github.com/flairNLP/zelda"
_URL = "https://nlp.informatik.hu-berlin.de/resources/datasets/zelda/zelda_full.zip"
_LABEL_DATA="https://raw.githubusercontent.com/flairNLP/VerbalizED/refs/heads/main/data/verbalizations/zelda_labels.txt"
_VERBALIZED_DATA="entity_linkings/entity_dictionary/zelda_wiki/zelda_labels_verbalizations.json"


logger = datasets.utils.logging.get_logger(__name__)


TEST_DATASET_NAME = [
    "aida-b",
    "tweeki",
    "reddit-comments",
    "reddit-posts",
    "wned-wiki",
    "cweb",
    "shadowlinks-top",
    "shadowlinks-tail",
    "shadowlinks-shadow"
]


class ZELDA_WIKI(GeneratorBasedBuilder):
    """
    Kensho Derived Wikimedia Dataset (ZELDA)
    """

    BUILDER_CONFIGS = [
        EntityDictionaryConfig(
            name=_DATASET_NAME,
            version=VERSION,
            description=_DESCRIPTION,
        ),
    ]

    def _info(self) -> DatasetInfo:
        return DatasetInfo(
            description=_DESCRIPTION,
            features=dictionary_features,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license="unknown",
        )

    def _split_generators(self, dl_manager: DownloadManager) -> list[SplitGenerator]:
        """Returns SplitGenerators."""
        data_dir = os.path.join(dl_manager.download_and_extract(_URL), "zelda")
        return [
            SplitGenerator(
                name=NamedSplit("dictionary"),
                gen_kwargs={"data_dir": data_dir},
            ),
        ]

    @staticmethod
    def get_all_titles_in_zelda(data_dir: str) -> dict[str, str]:
        """
        Convert a JSONL file to our specific format.
        """

        file_names = [os.path.join(data_dir, "train_data", "zelda_train.jsonl")]
        for dataset_name in TEST_DATASET_NAME:
            file_names.append(os.path.join(os.path.join(data_dir, "test_data", "jsonl", f"test_{dataset_name}.jsonl")))
        id_to_title = {}
        for file_name in file_names:
            with open(file_name, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line.strip())
                    wikipedia_titles = item.get("wikipedia_titles")
                    wikipedia_ids = item.get("wikipedia_ids")
                    assert len(wikipedia_ids) == len(wikipedia_titles)
                    for idx, title in zip(wikipedia_ids, wikipedia_titles):
                        if idx not in id_to_title:
                            id_to_title[idx] = title.replace(' ', '_')
        return id_to_title

    def _generate_examples(self, data_dir: str) -> Iterator[tuple[str, dict[str, Any]]]:
        """Generate examples for the specified split."""

        logger.info(f"Generating examples from {data_dir}")

        uid = map(str, itertools.count(start=0, step=1))
        titles = self.get_all_titles_in_zelda(data_dir)

        zelda_labels = [line.strip() for line in open(_LABEL_DATA, 'r')]
        zelda_verbalize = json.load(open(_VERBALIZED_DATA, 'r', encoding='utf-8'))

        title_to_id = {title: id for id, title in titles.items()}
        for zelda_label in zelda_labels:
            if zelda_label in zelda_verbalize:
                wiki_id = title_to_id[zelda_label]
                example = {
                    "dictionary": _DATASET_NAME,
                    "id": str(wiki_id),
                    "name": zelda_label,
                    "description": zelda_verbalize[zelda_label]['wikidata_description']
                }
                yield next(uid), example
            else:
                wiki_id = title_to_id[zelda_label]
                example = {
                    "dictionary": _DATASET_NAME,
                    "id": str(wiki_id),
                    "name": zelda_label,
                    "description": ""
                }
                yield next(uid), example

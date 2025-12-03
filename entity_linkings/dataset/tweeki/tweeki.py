import itertools
from typing import Any, Iterator

import datasets
from datasets import (
    DatasetInfo,
    DownloadManager,
    GeneratorBasedBuilder,
    Split,
    SplitGenerator,
)

from ..entity_linkikngs_hub import VERSION, EntityLinkingDatasetConfig, el_features
from ..utils import read_jsonl

_LOCAL = False
_LANGUAGES = ["English"]
_CITATION = """\
@inproceedings{tweeki:wnut20,
  author = {Bahareh Haradizadeh and Sameer Singh},
  title = { {Tweeki: Linking Named Entities on Twitter to a Knowledge Graph} },
  booktitle = {Workshop on Noisy User-generated Text (W-NUT)},
  year = {2020}
}
"""

_DATASET_NAME = "tweeki"
_DISPLAY_NAME = "TWEEKI"

_DESCRIPTION = """\
This dataset contains TWEEKI corpus"""

_HOMEPAGE = "https://ucinlp.github.io/tweeki/"
_URL = "https://raw.githubusercontent.com/ucinlp/tweeki/main/data/Tweeki_gold/Tweeki_gold.jsonl"


logger = datasets.utils.logging.get_logger(__name__)


class TWEEKI(GeneratorBasedBuilder):
    """
    TWEEKI dataset
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
        return [
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={"data_path": dl_manager.download_and_extract(_URL)},
            ),
        ]

    def _generate_examples(self, data_path: str) -> Iterator[tuple[str, dict[str, Any]]]:
        """"""
        logger.info(f"Generating examples from {data_path}")

        uid = map(str, itertools.count(start=0, step=1))
        for data in read_jsonl(data_path):
            entities = []
            for index, link in zip(data["index"], data["link"]):
                start, end = index
                _, qid = link.split('|')
                entities.append({"start": start, "end": end, "label": [qid]})

            example = {
                "dataset": _DATASET_NAME,
                "id": data["id"],
                "text": data["sentence"],
                "entities": entities,
            }

            yield next(uid), example

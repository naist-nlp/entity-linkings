import itertools
import os
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

_LOCAL = True
_LANGUAGES = ["English"]
_CITATION = """\
@inproceedings{DBLP:conf/aaai/OnoeD20,
    author       = {Yasumasa Onoe and Greg Durrett},
    title        = {Fine-Grained Entity Typing for Domain Independent Entity Linking},
    booktitle    = {The Thirty-Fourth {AAAI} Conference on Artificial Intelligence, {AAAI}
                    2020, The Thirty-Second Innovative Applications of Artificial Intelligence
                    Conference, {IAAI} 2020, The Tenth {AAAI} Symposium on Educational
                    Advances in Artificial Intelligence, {EAAI} 2020, New York, NY, USA,
                    February 7-12, 2020},
    pages        = {8576--8583},
    publisher    = {{AAAI} Press},
    year         = {2020},
    url          = {https://doi.org/10.1609/aaai.v34i05.6380},
    doi          = {10.1609/AAAI.V34I05.6380},
}
"""

_DATASET_NAME = "unseen"
_DISPLAY_NAME = "UNSEEN"

_DESCRIPTION = """\
This dataset contains WikilinksNED Unseen-Mentions corpus"""

_HOMEPAGE = "https://github.com/yasumasaonoe/ET4EL"
_URL = "entity_linkings/dataset/unseen/unseen_mentions.tar.gz"


logger = datasets.utils.logging.get_logger(__name__)


class UNSEEN(GeneratorBasedBuilder):
    """
    WikilinksNED Unseen-Mentions dataset
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
        data_dir = dl_manager.download_and_extract(_URL)
        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={"data_dir": data_dir, "split_id": "train"},
            ),
            SplitGenerator(
                name=Split.VALIDATION,
                gen_kwargs={"data_dir": data_dir, "split_id": "dev"},
            ),
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={"data_dir": data_dir, "split_id": "test"},
            ),
        ]

    def _generate_examples(self, data_dir: str, split_id: str) -> Iterator[tuple[str, dict[str, Any]]]:
        """"""
        logger.info(f"Generating examples from {data_dir}, split: {split_id}")
        data_path = os.path.join(data_dir, "unseen_mentions")

        if split_id == "train":
            data_path = os.path.join(data_path, "train")
            file_names = ["train_0.json", "train_1.json", "train_2.json", "train_3.json", "train_4.json", "train_5.json"]
        elif split_id == "dev":
            file_names = ["dev.json"]
        else:
            file_names = ["test.json"]

        uid = map(str, itertools.count(start=0, step=1))
        for file_name in file_names:
            file_path = os.path.join(data_path, file_name)
            raw_examples = read_jsonl(file_path)
            for data in raw_examples:
                context = " ".join([data["left_context_text"], data["word"]])
                end = len(context)
                start = end - len(data["word"])
                context = " ".join([context, data["right_context_text"]])

                eid = next(uid)
                example = {
                    "id": eid,
                    "text": context,
                    "entities": [{"start": start, "end": end, "label": [data["wikiId"]]}]
                }
                yield eid, example

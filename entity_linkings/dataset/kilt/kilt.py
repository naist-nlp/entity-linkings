import itertools
from typing import Any, Iterator

import datasets
from datasets import (
    Dataset,
    DatasetInfo,
    DownloadManager,
    GeneratorBasedBuilder,
    Split,
    SplitGenerator,
)

from ..entity_linkikngs_hub import VERSION, EntityLinkingDatasetConfig, el_features
from ..utils import read_jsonl

_LOCAL = False
_LANGUAGES = ["English'"]
_CITATION = """\
@inproceedings{petroni-etal-2021-kilt,
    title = "{KILT}: a Benchmark for Knowledge Intensive Language Tasks",
    author = {Petroni, Fabio  and Piktus, Aleksandra  and
      Fan, Angela  and Lewis, Patrick  and
      Yazdani, Majid  and De Cao, Nicola  and
      Thorne, James  and Jernite, Yacine  and
      Karpukhin, Vladimir  and Maillard, Jean  and
      Plachouras, Vassilis  and Rockt{\"a}schel, Tim  and
      Riedel, Sebastian},
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association 
                 for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.naacl-main.200",
    doi = "10.18653/v1/2021.naacl-main.200",
    pages = "2523--2544",
}
"""

_DATASET_NAME = "kilt"
_DISPLAY_NAME = "KILT"

_DESCRIPTION = """\
This dataset contains KILT corpus, a comprehensive benchmark for supervised entity disambiguation.
"""

_HOMEPAGE = "https://github.com/flairNLP/zelda"
_URLs = [
    "http://dl.fbaipublicfiles.com/KILT/blink-train-kilt.jsonl",
    "http://dl.fbaipublicfiles.com/KILT/blink-dev-kilt.jsonl"
]

logger = datasets.utils.logging.get_logger(__name__)


class KILT(GeneratorBasedBuilder):
    """
    KILT dataset
    """

    BUILDER_CONFIGS = [
        EntityLinkingDatasetConfig(
            name="wned_wiki",
            version=VERSION,
            description=_DESCRIPTION,
            subset_id='wned_wiki',
        ),
        EntityLinkingDatasetConfig(
            name="wned_cweb",
            version=VERSION,
            description=_DESCRIPTION,
            subset_id='wned_cweb',
        ),
        EntityLinkingDatasetConfig(
            name="aidayago2",
            version=VERSION,
            description=_DESCRIPTION,
            subset_id='aidayago2',
        ),
        EntityLinkingDatasetConfig(
            name="wikipedia",
            version=VERSION,
            description=_DESCRIPTION,
            subset_id='wikipedia',
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

        if self.config.subset_id == 'wikipedia':
            return [
                SplitGenerator(
                    name=Split.TRAIN,
                    gen_kwargs={"dataset": dl_manager.download_and_extract(_URLs[0])},
                ),
                SplitGenerator(
                    name=Split.VALIDATION,
                    gen_kwargs={"dataset": dl_manager.download_and_extract(_URLs[1])},
                ),
            ]
        elif self.config.subset_id == 'aidayago2':
            dataset = datasets.load_dataset("facebook/kilt_tasks", "aidayago2", trust_remote_code=True)
            return [
                SplitGenerator(
                    name=Split.TRAIN,
                    gen_kwargs={"dataset": dataset["train"]},
                ),
                SplitGenerator(
                    name=Split.VALIDATION,
                    gen_kwargs={"dataset": dataset["validation"]},
                ),
                SplitGenerator(
                    name=Split.TEST,
                    gen_kwargs={"dataset": dataset["test"]},
                ),
            ]
        else:
            dataset = datasets.load_dataset("facebook/kilt_tasks", "aidayago2", trust_remote_code=True)
            return [
                SplitGenerator(
                    name=Split.VALIDATION,
                    gen_kwargs={"dataset": dataset["validation"]},
                ),
                SplitGenerator(
                    name=Split.TEST,
                    gen_kwargs={"dataset": dataset["test"]},
                ),
            ]

    def _generate_examples(self, dataset: str | Dataset) -> Iterator[tuple[str, dict[str, Any]]]:
        """Yields examples."""
        logger.info("Generating examples from dataset")

        if isinstance(dataset, str):
            dataset = read_jsonl(dataset)

        uid = map(str, itertools.count(start=0, step=1))
        for data in dataset:
            mention = data["meta"]["mention"].replace(u'\xa0', ' ')
            left_context = data["meta"]["left_context"].replace(u'\xa0', ' ')
            right_context = data["meta"]["right_context"].replace(u'\xa0', ' ')

            text = f"{left_context} {mention} {right_context}".strip()
            start = len(left_context) + 1 if left_context else 0
            end = start + len(mention)

            assert text[start: end] == mention, (text[start: end], mention)
            if data['output'] != []:
                assert len(data['output']) == 1 and len(data['output'][0]['provenance']) == 1
                wikipedia_id = f"Q{data['output'][0]['provenance'][0]['wikipedia_id']}"
            else:
                wikipedia_id = "-1"

            yield next(uid), {
                "id": data["id"],
                "text": text,
                "entities": [{
                    "start": start,
                    "end": end,
                    "label": [wikipedia_id],
                }],
            }

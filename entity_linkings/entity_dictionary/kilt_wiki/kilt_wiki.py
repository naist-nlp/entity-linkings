import itertools
import json
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

_DESCRIPTION = """\
This dataset contains KILT's knowledge source
"""

_HOMEPAGE = "https://ai.meta.com/tools/kilt/"
_URL='entity_linkings/entity_dictionary/kilt_wiki/kilt_knowledgesource.json'

logger = datasets.utils.logging.get_logger(__name__)
_LICENCE = "unknown"


class KILT_WIKI(GeneratorBasedBuilder):
    """
    KILT knowledge source
    """

    BUILDER_CONFIGS = [
        EntityDictionaryConfig(
            name=_DATASET_NAME,
            version=VERSION,
            description=_DESCRIPTION,
        )
    ]

    def _info(self) -> DatasetInfo:
        return DatasetInfo(
            description=_DESCRIPTION,
            features=dictionary_features,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENCE,
        )

    def _split_generators(self, dl_manager: DownloadManager) -> list[SplitGenerator]:
        """Returns SplitGenerators."""
        return [
            SplitGenerator(
                name=NamedSplit("dictionary"),
                gen_kwargs={"input_path": dl_manager.download_and_extract(_URL)},
            ),
        ]

    def _generate_examples(self, input_path: str) -> Iterator[tuple[str, dict[str, Any]]]:
        """Yields examples."""
        logger.info("Generating examples from dataset")

        uid = map(str, itertools.count(start=0, step=1))
        with open(input_path, encoding="utf-8") as f:
            for line in f:
                line = json.loads(line.strip())
                description = "" if len(line['text']) < 2 else line['text'][1].strip()
                entity = {
                    "dictionary": _DATASET_NAME,
                    "id": f"{line['wikipedia_id']}",
                    "name": line['wikipedia_title'],
                    "description": description
                }
                yield next(uid), entity

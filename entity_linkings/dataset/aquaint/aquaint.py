"https://www.nzdl.org/wikification/data/wikifiedStories.zip'"

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
from ..utils import get_wikipedia_summary, read_text_file

_LOCAL = False
_LANGUAGES = ["English"]
_CITATION = """\
@inproceedings{milne-witten-2008-aquaint,
    title = "Learning to link with wikipedia",
    author = "Milne, David and Witten, Ian H.",
    booktitle = "Proceedings of the 17th ACM Conference on Information and Knowledge Management",
    year = "2008",
    series = {CIKM '08},
    address = "New York, NY, USA",
    publisher = "Association for Computing Machinery",
    url = "https://doi.org/10.1145/1458082.1458150",
    doi = "10.1145/1458082.1458150",
    pages = "509–-518",
}
"""

_DATASET_NAME = "aquaint"
_DISPLAY_NAME = "AQUAINT"

_DESCRIPTION = """\
This dataset contains AQUAINT corpus.
"""

_HOMEPAGE = "https://community.nzdl.org/wikification/"
_URL = "https://community.nzdl.org/wikification/data/wikifiedStories.zip"
_LICENCE = "GNU Public Licence"

logger = datasets.utils.logging.get_logger(__name__)


class AQUAINT(GeneratorBasedBuilder):
    """
    AQUAINT dataset
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
                name=Split.TEST,
                gen_kwargs={"data_dir": data_dir},
            ),
        ]

    def _generate_examples(self, data_dir: str) -> Iterator[tuple[str, dict[str, Any]]]:
        """"""
        logger.info(f"Generating examples from {data_dir}")
        file_names = [fn for fn in os.listdir(data_dir) if fn.endswith('.htm')]
        entity_dict = {}

        uid = map(str, itertools.count(start=0, step=1))
        for i, file_name in enumerate(file_names):
            raw_text = read_text_file(os.path.join(data_dir, file_name))
            texts = raw_text.split('<p> ')
            texts[0] = texts[0].split('<h1 id="header">')[1][:-7]
            for i in range(1, len(texts) - 1):
                texts[i] = texts[i][:-7]
            texts[-1] = texts[-1][:-23]

            for text in texts:
                if not text.strip():
                    continue

                entities = []
                current_entity = text.find("[[")
                while current_entity != -1:
                    wikiname = ""
                    surface = ""
                    j = current_entity + 2

                    while text[j] not in ["]", "|"]:
                        wikiname += text[j]
                        j += 1

                    if text[j] == "]":
                        surface = wikiname
                    else:
                        j += 1
                        while text[j] not in ["]", "|"]:
                            surface += text[j]
                            j += 1

                        if text[j] =="|":
                            agreement_score = float(text[j+1: j+4])
                            j += 4
                            if agreement_score < 0.5:
                                text = text[:current_entity] + surface + text[j+2:]
                                current_entity = text.find("[[")
                                continue

                    title = wikiname[0].upper() + wikiname.replace(" ", "_")[1:]
                    if title not in entity_dict:
                        wiki_summary = get_wikipedia_summary(title)
                        entity_dict[title] = wiki_summary
                    pageid = entity_dict[title]['pageid']
                    entities.append({
                        "start": current_entity,
                        "end": j + 1,
                        "label": [str(pageid)],
                    })

                    text = text[:current_entity] + surface + text[j+2:]
                    current_entity = text.find("[[")

                pid = next(uid)
                example = {
                    "dataset": "aquaint",
                    "id": file_name[:-4]+"-"+str(pid),
                    "text": text,
                    "entities": entities,
                }
                yield pid, example

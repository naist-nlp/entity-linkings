"https://www.nzdl.org/wikification/data/wikifiedStories.zip'"

import itertools
import os
from typing import Any, Iterator
from xml.etree import ElementTree

import datasets
from datasets import (
    DatasetInfo,
    DownloadManager,
    GeneratorBasedBuilder,
    Split,
    SplitGenerator,
)

from ..entity_linkikngs_hub import VERSION, EntityLinkingDatasetConfig, el_features
from ..utils import get_wikipedia_summary

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

_DATASET_NAME = "msnbc"
_DISPLAY_NAME = "MSNBC"

_DESCRIPTION = """\
This dataset contains MSNBC corpus.
"""

_HOMEPAGE = "http://research.microsoft.com/en-us/um/people/silviu/WebAssistant/TestData/"
_URL = "https://github.com/dice-group/gerbil/releases/download/v1.2.6/gerbil_data.zip"
_LICENCE = "unknown"

logger = datasets.utils.logging.get_logger(__name__)


class MSNBC(GeneratorBasedBuilder):
    """
    MSNBC dataset
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
        data_dir = os.path.join(dl_manager.download_and_extract(_URL), "gerbil_data", "datasets", "MSNBC")
        return [
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={"data_dir": data_dir},
            ),
        ]

    @staticmethod
    def get_documents(data_dir: str) -> dict[str, str]:
        file_names = os.listdir(data_dir)
        documents = {}
        for file_name in file_names:
            doc_id = file_name.split(".")[0]
            with open(os.path.join(data_dir, file_name), 'r', encoding='utf-8') as f:
                documents[doc_id] = f.read()
        return documents

    @staticmethod
    def parse_annotation(file_path: str) -> list[dict[str, Any]]:
        entities = []
        with open(file_path, 'r', encoding='utf-8') as f:
            g = ElementTree.parse(f)
            root = g.getroot()
            for anno in root.findall('ReferenceInstance'):
                start = anno.findtext('Offset')
                length = anno.findtext('Length')
                wikilink = anno.findtext('ChosenAnnotation')
                surface_form = anno.findtext('SurfaceForm')
                if start is None or length is None or wikilink is None or surface_form is None:
                    raise ValueError("Missing required annotation fields.")

                wiki_summary = get_wikipedia_summary(wikilink.split('/')[-1].strip())
                entities.append({
                    "start": int(start),
                    "end": int(start) + int(length),
                    "label": [wiki_summary['pageid']],
                })
        return entities

    def _generate_examples(self, data_dir: str) -> Iterator[tuple[str, dict[str, Any]]]:
        """"""
        logger.info(f"Generating examples from {data_dir}")

        raw_documents = self.get_documents(os.path.join(data_dir, "RawTextsSimpleChars_utf8"))

        uid = map(str, itertools.count(start=0, step=1))
        for kid, content in raw_documents.items():
            entities = self.parse_annotation(os.path.join(data_dir, "Problems", f"{kid}.txt"))
            example = {"dataset": "msnbc", "id": kid, "text": content, "entities": entities}
            yield next(uid), example



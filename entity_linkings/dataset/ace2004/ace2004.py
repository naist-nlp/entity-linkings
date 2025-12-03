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
@inproceedings{ratinov-etal-2011-local,
    title = "Local and Global Algorithms for Disambiguation to {W}ikipedia",
    author = "Ratinov, Lev  and
        Roth, Dan  and
        Downey, Doug  and
        Anderson, Mike",
    booktitle = "Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2011",
    address = "Portland, Oregon, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P11-1138/",
    pages = "1375--1384"
}
"""

_DATASET_NAME = "ace2004"
_DISPLAY_NAME = "ACE2004"

_DESCRIPTION = """\
This dataset contains ACE2004 corpus.
"""

_HOMEPAGE = "https://cogcomp.seas.upenn.edu/page/resource_view/4"
# _URL = "http://cogcomp.seas.upenn.edu/Data/ACL2011WikificationData.zip"
_URL = "https://github.com/dice-group/gerbil/releases/download/v1.2.6/gerbil_data.zip"
_LICENCE = "unknown"

logger = datasets.utils.logging.get_logger(__name__)


class ACE2004(GeneratorBasedBuilder):
    """
    ACE2004 dataset
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
        # data_dir = os.path.join(dl_manager.download_and_extract(_URL), "WikificationACL2011Data", "ACE2004_Coref_Turking", "Dev")
        data_dir = os.path.join(dl_manager.download_and_extract(_URL), "gerbil_data", "datasets", "ACE2004_Coref_Turking", "Dev")
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
            if '.svn' in file_name:
                continue
            with open(os.path.join(data_dir, file_name), 'r', encoding='utf-8') as f:
                documents[file_name] = f.read()
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

        raw_documents = self.get_documents(os.path.join(data_dir, "RawTextsNoTranscripts"))

        uid = map(str, itertools.count(start=0, step=1))
        for kid, content in raw_documents.items():
            entities = self.parse_annotation(os.path.join(data_dir, "ProblemsNoTranscripts", f"{kid}"))
            example = {"dataset": "ace2004", "id": kid, "text": content, "entities": entities}
            yield next(uid), example



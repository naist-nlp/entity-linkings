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

_LOCAL = True
_LANGUAGES = ["English"]
_CITATION = """\
@inproceedings{DBLP:conf/cikm/GuoB14,
    author    = {Zhaochen Guo and Denilson Barbosa},
    title     = {Robust Entity Linking via Random Walks},
    booktitle = {Proceedings of the 23rd {ACM} International Conference on Conference
                on Information and Knowledge Management, {CIKM} 2014, Shanghai, China,
                November 3-7, 2014},
    pages     = {499--508},
    year      = {2014},
    url       = {http://doi.acm.org/10.1145/2661829.2661887},
    doi       = {10.1145/2661829.2661887}
}
"""

_DATASET_NAME = "wned"
_DISPLAY_NAME = "WNED"

_DESCRIPTION = """\
This dataset contains WNED corpus.
"""

_HOMEPAGE = "https://github.com/U-Alberta/wned"
_URL = "entity_linkings/dataset/wned/WNED.tar.gz"
_LICENCE = "Unknown"

logger = datasets.utils.logging.get_logger(__name__)

class WNED(GeneratorBasedBuilder):
    """
    WNED dataset
    """

    BUILDER_CONFIGS = [
        EntityLinkingDatasetConfig(
            name="wiki",
            version=VERSION,
            description=_DESCRIPTION,
            subset_id='wiki',
        ),
        EntityLinkingDatasetConfig(
            name="cweb",
            version=VERSION,
            description=_DESCRIPTION,
            subset_id='cweb',
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
        data_dir = os.path.join(dl_manager.download_and_extract(_URL), "WNED", "wned-datasets")
        if self.config.subset_id == 'wiki':
            return [
                SplitGenerator(
                    name=Split.TEST,
                    gen_kwargs={"data_dir": data_dir, "subset_id": "wikipedia"},
                ),
            ]
        else:
            return [
                SplitGenerator(
                    name=Split.TEST,
                    gen_kwargs={"data_dir": data_dir, "subset_id": "clueweb"},
                ),
            ]

    def get_documents(self, data_dir: str) -> dict[str, str]:
        documents = {}
        files = os.path.join(data_dir, "RawText")
        for fn in os.listdir(files):
            file_path = os.path.join(files, fn)
            with open(file_path, encoding="utf-8") as f:
                documents[fn] = f.read()
        return documents

    def _generate_examples(self, data_dir: str, subset_id: str) -> Iterator[tuple[str, dict[str, Any]]]:
        """"""
        logger.info(f"Generating examples from {data_dir} for subset: {subset_id}")
        data_dir = os.path.join(data_dir, subset_id if subset_id == "wikipedia" else "clueweb12")
        raw_texts = self.get_documents(data_dir)
        entity_dict = {}

        xml_file = os.path.join(data_dir, f"{subset_id}.xml")
        tree = ElementTree.parse(xml_file)
        root = tree.getroot()

        uid = map(str, itertools.count(start=0, step=1))
        documents = root.findall("document")
        for elem in documents:
            entities = []
            docname = elem.attrib['docName']
            for child in elem.findall("annotation"):
                wikiname = child[1].text.replace(' ', '_')
                start = int(child[2].text)
                end = start + int(child[3].text)
                if wikiname not in entity_dict:
                    wiki_summary = get_wikipedia_summary(wikiname)
                    entity_dict[wikiname] = wiki_summary
                pageid = str(entity_dict[wikiname]['pageid'])
                assert child[0].text == raw_texts[docname][start:end], f"{child[0].text} != {raw_texts[docname][start:end]}"
                entities.append({"start": start, "end": end, "label": [pageid]})

            example = {
                "name": f"wned-{subset_id}",
                "id": docname,
                "text": raw_texts[docname],
                "entities": entities,
            }
            yield next(uid), example

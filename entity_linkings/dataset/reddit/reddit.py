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
from ..utils import get_wikipedia_summary

_LOCAL = False
_LANGUAGES = ["English"]
_CITATION = """\
@article{10.1016/j.ipm.2020.102479,
    author = {Botzer, Nicholas and Ding, Yifan and Weninger, Tim},
    title = {Reddit entity linking dataset},
    year = {2021},
    issue_date = {May 2021},
    publisher = {Pergamon Press, Inc.},
    address = {USA},
    volume = {58},
    number = {3},
    issn = {0306-4573},
    url = {https://doi.org/10.1016/j.ipm.2020.102479},
    doi = {10.1016/j.ipm.2020.102479},
    journal = {Inf. Process. Manage.},
    month = may,
    numpages = {13},
    keywords = {Entity linking, Dataset, Natural language processing}
}
"""

_DATASET_NAME = "reddit"
_DISPLAY_NAME = "REDDIT"

_DESCRIPTION = """\
This dataset contains REDDIT corpus"""

_HOMEPAGE = "hhttps://zenodo.org/records/3970806"
_URL = "https://zenodo.org/records/3970806/files/reddit_el.zip"


logger = datasets.utils.logging.get_logger(__name__)


class REDDIT(GeneratorBasedBuilder):
    """
    REDDIT dataset
    """

    BUILDER_CONFIGS = [
        EntityLinkingDatasetConfig(
            name="posts",
            version=VERSION,
            description=_DESCRIPTION,
            subset_id="posts",
        ),
        EntityLinkingDatasetConfig(
            name="comments",
            version=VERSION,
            description=_DESCRIPTION,
            subset_id="comments",
        )
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
        if self.config.subset_id == "posts":
            text_path = os.path.join(dl_manager.download_and_extract(_URL), "posts.tsv")
            anno_path = os.path.join(dl_manager.download_and_extract(_URL), "gold_post_annotations.tsv")
            return [
                SplitGenerator(
                    name=Split.TEST,
                    gen_kwargs={"text_path": text_path, "anno_path": anno_path, "subset_id": self.config.subset_id},
                ),
            ]
        else:
            text_path = os.path.join(dl_manager.download_and_extract(_URL), "comments.tsv")
            anno_path = os.path.join(dl_manager.download_and_extract(_URL), "gold_comment_annotations.tsv")
            return [
                SplitGenerator(
                    name=Split.TEST,
                    gen_kwargs={"text_path": text_path, "anno_path": anno_path, "subset_id": self.config.subset_id},
                ),
            ]

    @staticmethod
    def get_documents(input_path: str, id_col: int, text_col: int) -> dict[str, str]:
        documents: dict[str, str] = {}
        post_id = "-1"
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                cols = line.split('\t')
                if len(cols) == 1:
                    documents[post_id] += cols[0]
                    continue
                doc_id = cols[id_col]
                documents[doc_id] = cols[text_col]
                post_id = doc_id
        return documents

    @staticmethod
    def get_annotation(input_path: str) -> dict[str, Any]:
        annotations: dict[str, Any] = {}
        lines = open(input_path, 'r', encoding='utf-8').readlines()
        for line in lines:
            if line.strip():
                cols = line.strip().split('\t')
                doc_id = cols[0]
                mention = cols[2]
                wikiname = cols[3]
                start = int(cols[4])
                end = int(cols[5])
                if doc_id not in annotations:
                    annotations[doc_id] = []
                annotations[doc_id].append({
                    "mention": mention,
                    "start": start,
                    "end": end,
                    "wikiname": wikiname,
                })
        return annotations

    def _generate_examples(self, text_path: str, anno_path: str, subset_id: str) -> Iterator[tuple[str, dict[str, Any]]]:
        """"""
        logger.info(f"Generating examples from {text_path} and {anno_path} for subset: {subset_id}")

        if subset_id == "posts":
            raw_examples = self.get_documents(text_path, id_col=0, text_col=2)
        else:
            raw_examples = self.get_documents(text_path, id_col=0, text_col=4)
        annotations = self.get_annotation(anno_path)
        entity_dict = {}

        uid = map(str, itertools.count(start=0, step=1))
        for doc_id, content in raw_examples.items():
            entities = []
            content = content.strip()
            for entity in annotations.get(doc_id, []):
                wikiname = entity["wikiname"]
                if wikiname not in entity_dict:
                    wiki_summary = get_wikipedia_summary(wikiname)
                    entity_dict[wikiname] = wiki_summary
                wiki_id = str(entity_dict[wikiname]['pageid'])
                assert content[entity["start"]:entity["end"]] == entity["mention"], f"{content[entity['start']:entity['end']]} != {entity['mention']}"

                entities.append({
                    "start": entity["start"],
                    "end": entity["end"],
                    "label": [wiki_id],
                })

            example = {
                "dataset": "reddit_posts",
                "id": doc_id,
                "text": content,
                "entities": entities,
            }

            yield next(uid), example

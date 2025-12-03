import itertools
import json
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
@inproceedings{10.1145/3539618.3591912,
    title = {{Linked-DocRED – Enhancing DocRED with Entity-Linking to Evaluate End-To-End Document-Level Information Extraction Pipelines}},
    booktitle = {{Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR'23)}},
    author = {Genest, Pierre-Yves and Portier, Pierre-Edouard and Egyed-Zsigmond, El\"{o}d and Lovisetto, Martino},
    year = {2023},
    pages = {11},
    publisher = {{Association for Computing Machinery}},
    address = {{Taipei, Taiwan}},
    doi = {10.1145/3539618.3591912},
    isbn = {978-1-4503-9408-6},
}
"""

_DATASET_NAME = "linked-re-docred"
_DISPLAY_NAME = "Linked-Re-DOCRED"

_DESCRIPTION = """\
This dataset contains Linked-Re-DOCRED corpus"""

_HOMEPAGE = "https://github.com/alteca/Linked-DocRED/tree/main/Linked-Re-DocRED"
_URLs = {
    "train": "https://raw.githubusercontent.com/alteca/Linked-DocRED/refs/heads/main/Linked-Re-DocRED/train_revised.json",
    "validation": "https://raw.githubusercontent.com/alteca/Linked-DocRED/refs/heads/main/Linked-Re-DocRED/dev_revised.json",
    "test": "https://raw.githubusercontent.com/alteca/Linked-DocRED/refs/heads/main/Linked-Re-DocRED/test_revised.json"
}


logger = datasets.utils.logging.get_logger(__name__)


class DOCRED(GeneratorBasedBuilder):
    """
    SHADOWLINK dataset
    """

    BUILDER_CONFIGS = [
        EntityLinkingDatasetConfig(
            name=_DATASET_NAME,
            version=VERSION,
            description=_DESCRIPTION,
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
                name=Split.TRAIN,
                gen_kwargs={
                    "input_path": dl_manager.download_and_extract(_URLs[Split.TRAIN]),
                },
            ),
            SplitGenerator(
                name=Split.VALIDATION,
                gen_kwargs={
                    "input_path": dl_manager.download_and_extract(_URLs[Split.VALIDATION]),
                },
            ),
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={
                    "input_path": dl_manager.download_and_extract(_URLs[Split.TEST]),
                },
            ),
        ]

    def _generate_examples(self, input_path: str) -> Iterator[tuple[str, dict[str, Any]]]:
        """"""
        logger.info(f"Generating examples from {input_path}")
        raw_examples = json.load(open(input_path, 'r', encoding='utf-8'))
        entity_dict = {}

        uid = map(str, itertools.count(start=0, step=1))
        for i, r_e in enumerate(raw_examples):
            sentences = r_e['sents']
            entities: list[list[dict[str, Any]]] = [[] for _ in sentences]
            for entity in r_e['entities']:
                entity_info = entity['entity_linking']
                if entity_info['confidence'] != 'A':
                    continue
                wikipedia_resource = entity_info['wikipedia_resource']
                if wikipedia_resource in ['#ignored#', f"#DocRED-{entity['id']}"]:
                    page_id = "-1"
                else:
                    if wikipedia_resource not in entity_dict:
                        wiki_summary = get_wikipedia_summary(wikipedia_resource)
                        entity_dict[wikipedia_resource] = str(wiki_summary['pageid'])
                    page_id = entity_dict[wikipedia_resource]
                for mention in entity["mentions"]:
                    sent_id = mention['sent_id']
                    start, end = mention['pos'][0], mention['pos'][1]
                    char_start = len(" ".join(sentences[sent_id][:start])) + (1 if start > 0 else 0)
                    char_end = len(" ".join(sentences[sent_id][:end]))
                    entities[sent_id].append({
                        "start": char_start,
                        "end": char_end,
                        "label": [page_id],
                    })
            for i, sent in enumerate(sentences):
                example = {
                    "dataset": "linked_re-docred",
                    "id": f"{r_e['title']}_{i}",
                    "text": " ".join(sent),
                    "entities": entities[i],
                }
                yield next(uid), example

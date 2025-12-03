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
from ..utils import read_conll

_LOCAL = False
_LANGUAGES = ["English"]
_CITATION = """\
@inproceedings{milich2023zelda,
    title={{ZELDA}: A Comprehensive Benchmark for Supervised Entity Disambiguation},
    author={Milich, Marcel and Akbik, Alan},
    booktitle={{EACL} 2023,  The 17th Conference of the European Chapter of the Association for Computational Linguistics},
    year={2023}
}
"""

_DATASET_NAME = "zelda"
_DISPLAY_NAME = "ZELDA"

_DESCRIPTION = """\
This dataset contains ZELDA corpus, a comprehensive benchmark for supervised entity disambiguation.
"""

_HOMEPAGE = "https://github.com/flairNLP/zelda"
_URL = "https://nlp.informatik.hu-berlin.de/resources/datasets/zelda/zelda_full.zip"


logger = datasets.utils.logging.get_logger(__name__)


TEST_DATASET_NAME = [
    "aida-b",
    "tweeki",
    "reddit-comments",
    "reddit-posts",
    "wned-wiki",
    "cweb",
    "shadowlinks-top",
    "shadowlinks-tail",
    "shadowlinks-shadow"
]


class ZELDA(GeneratorBasedBuilder):
    """
    ZELDA dataset
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
        data_dir = os.path.join(dl_manager.download_and_extract(_URL), "zelda")
        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={"data_dir": data_dir, "split": "train"},
            ),
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={"data_dir": data_dir, "split": "test"},
            ),
        ]

    def _concat_sentence(self, example: list[dict[str, Any]],
                        max_characters: int = 2800,
                        max_spans_per_sentence: int = 100,
                    ) -> list[dict[str, Any]]:

        char_counter, span_counter = 0, 0
        sentences: list[str] = []
        entities: list[dict[str, Any]] = []
        chunk_data: list[dict[str, Any]] = []
        for e in example:
            if char_counter + len(e["text"]) > max_characters or span_counter + len(e["entities"]) > max_spans_per_sentence:
                if sentences:
                    chunk_data.append({"text": " ".join(sentences), "entities": entities})
                char_counter, span_counter = 0, 0
                sentences, entities = [], []
            sentences.append(e["text"])
            _entities = []
            for ent in e["entities"]:
                start = ent["start"] + char_counter
                end = ent["end"] + char_counter
                assert " ".join(sentences)[start: end] == ent["text"], f"Entity text mismatch: '{' '.join(sentences)[start: end]}' != '{ent['text']}'"
                _entities.append({"start": start, "end": end, "label": ent["label"], "title": ent["title"], "text": ent["text"]})
            entities.extend(_entities)
            char_counter += len(e["text"]) + 1
            span_counter += len(e["entities"])
        if sentences:
            chunk_data.append({"text": " ".join(sentences), "entities": entities})

        return chunk_data

    def _generate_examples(self, data_dir: str, split: str) -> Iterator[tuple[str, dict[str, Any]]]:
        """"""
        logger.info(f"Generating examples for split: {split} from {data_dir}")

        if split == "train":
            dataset_names = ["zelda"]
            file_names = [os.path.join(data_dir, f"{split}_data", "zelda_train.conll")]
        else:
            dataset_names = TEST_DATASET_NAME
            file_names = [
                os.path.join(data_dir, f"{split}_data", "conll", f"test_{dataset_name}.conll")
                for dataset_name in TEST_DATASET_NAME
            ]

        uid = map(str, itertools.count(start=0, step=1))
        for i, file_name in enumerate(file_names):
            for j, (id, sentences) in enumerate(read_conll(file_name, delimiter='\t')):
                splitted_sentences = self._concat_sentence(sentences)
                id = str(j) if not id else str(id)
                for si, sentence in enumerate(splitted_sentences):
                    text = sentence['text']
                    entities = sentence['entities']
                    example = {
                        "dataset": dataset_names[i],
                        "id": f"{id}-{si}",
                        "text": text,
                        "entities": entities,
                    }
                    yield next(uid), example

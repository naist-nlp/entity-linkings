"https://www.nzdl.org/wikification/data/wikifiedStories.zip'"

import itertools
import os
from typing import Any, Iterable, Iterator, Union

import datasets
from datasets import (
    DatasetInfo,
    DownloadManager,
    GeneratorBasedBuilder,
    Split,
    SplitGenerator,
)

from ..entity_linkikngs_hub import VERSION, EntityLinkingDatasetConfig, el_features
from ..utils import _conll_to_example, get_wikipedia_summary

_LOCAL = False
_LANGUAGES = ["English"]
_CITATION = """\
@article{derczynski2015analysis,
  title={Analysis of named entity recognition and linking for tweets},
  author={Derczynski, Leon and Maynard, Diana and Rizzo, Giuseppe and Van Erp, Marieke and Gorrell, Genevieve and Troncy, Rapha{\"e}l and Petrak, Johann and Bontcheva, Kalina},
  journal={Information Processing \& Management},
  volume={51},
  number={2},
  pages={32--49},
  year={2015},
  publisher={Elsevier}
}
"""

_DATASET_NAME = "kore50"
_DISPLAY_NAME = "KORE50"

_DESCRIPTION = """\
This dataset contains KORE50 corpus.
"""

_HOMEPAGE = "https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/ambiverse-nlu/aida/"
_URL = "http://resources.mpi-inf.mpg.de/yago-naga/aida/download/KORE50.tar.gz"
_LICENCE = "CC BY-SA 3.0"

logger = datasets.utils.logging.get_logger(__name__)

class KORE50(GeneratorBasedBuilder):
    """
    KORE50 dataset
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
        data_file = os.path.join(data_dir, "KORE50", "AIDA.tsv")
        return [
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={"data_file": data_file},
            ),
        ]

    @staticmethod
    def read_conll(
            file: Union[str, bytes, os.PathLike],
            delimiter: str = ' ',
            word_column: int = 0,
            tag_column: int = 1,
            link_column: int = 2
        ) -> Iterable[list[dict[str, Any]]]:
        sentences: list[dict[str, Any]] = []
        words: list[str] = []
        labels: list[str] = []
        links: list[str] = []

        with open(file, mode="r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip()
                if line.startswith("-DOCSTART-"):
                    if sentences:
                        yield sentences
                        sentences = []
                elif not line:
                    if words:
                        sentences.append(_conll_to_example(words, labels, links))
                        words = []
                        labels = []
                        links = []
                else:
                    cols = line.split(delimiter)
                    if len(cols) == 1:
                        words.append(cols[word_column])
                        labels.append("O")
                        links.append("")
                    else:
                        words.append(cols[word_column])
                        labels.append(cols[tag_column])
                        links.append(cols[link_column])
            if words:
                sentences.append(_conll_to_example(words, labels, links))
            if sentences:
                yield sentences

    def _generate_examples(self, data_file: str) -> Iterator[tuple[str, dict[str, Any]]]:
        """"""
        logger.info(f"Generating examples from {data_file}")
        entity_dict = {}

        uid = map(str, itertools.count(start=0, step=1))
        for sentences in self.read_conll(data_file, delimiter='\t', tag_column=1, link_column=3):
            for example in sentences:
                entities = []
                for ent in example['entities']:
                    title  = ent['title'][0]
                    if title == '--NME--':
                        entities.append({"start": ent['start'], "end": ent['end'], "label": ["-1"]})
                        continue
                    if title not in entity_dict:
                        entity_dict[title] = get_wikipedia_summary(title)
                    entities.append({
                        "start": ent['start'],
                        "end": ent['end'],
                        "label": [str(entity_dict[title]['pageid'])],
                    })

                eid = next(uid)
                yield eid, {"dataset": "KORE50", "id": eid, "text": example['text'], "entities": entities}

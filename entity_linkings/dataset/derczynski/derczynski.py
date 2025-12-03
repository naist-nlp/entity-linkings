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
from ..utils import get_wikipedia_summary, read_conll

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

_DATASET_NAME = "derczynski"
_DISPLAY_NAME = "DERCZYNSKI"

_DESCRIPTION = """\
This dataset contains DERCZYNSKI corpus.
"""

_HOMEPAGE = "https://huggingface.co/datasets/strombergnlp/ipm_nel"
_URL = "http://www.derczynski.com/resources/ipm_nel.tar.gz"
_LICENCE = "CC-BY 4.0"

logger = datasets.utils.logging.get_logger(__name__)

class DERCZYNSKI(GeneratorBasedBuilder):
    """
    DERCZYNSKI dataset
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
        data_file = os.path.join(data_dir, "ipm_nel_corpus", "ipm_nel.conll")
        return [
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={"data_file": data_file},
            ),
        ]

    def _generate_examples(self, data_file: str) -> Iterator[tuple[str, dict[str, Any]]]:
        """"""
        logger.info(f"Generating examples from {data_file}")
        entity_dict = {}

        uid = map(str, itertools.count(start=0, step=1))
        for _, sentences in read_conll(data_file, delimiter='\t', tag_column=2, link_column=1):
            for sentence in sentences:
                entities = []
                for ent in sentence['entities']:
                    if ent['title'][0] == 'NIL':
                        entities.append({"start": ent['start'], "end": ent['end'], "label": ["-1"]})
                        continue
                    title = ent['title'][0].split('/')[-1]
                    if title not in entity_dict:
                        entity_dict[title] = get_wikipedia_summary(title)
                    entities.append({
                        "start": ent['start'],
                        "end": ent['end'],
                        "label": [str(entity_dict[title]['pageid'])],
                    })
                example_id = next(uid)
                example = {
                    "id": example_id,
                    "text": sentence['text'],
                    "entities": entities,
                }
                yield example_id, example

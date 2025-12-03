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

_LOCAL = False
_LANGUAGES = ["English"]
_CITATION = """\
@inproceedings{provatorova-etal-2021-robustness,
    title = "Robustness Evaluation of Entity Disambiguation Using Prior Probes: the Case of Entity Overshadowing",
    author = "Provatorova, Vera  and
        Bhargav, Samarth  and
        Vakulenko, Svitlana  and
        Kanoulas, Evangelos",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.820/",
    doi = "10.18653/v1/2021.emnlp-main.820",
    pages = "10501--10510",
    abstract = "Entity disambiguation (ED) is the last step of entity linking (EL), when candidate entities are reranked according to the context they appear in. All datasets for training and evaluating models for EL consist of convenience samples, such as news articles and tweets, that propagate the prior probability bias of the entity distribution towards more frequently occurring entities. It was shown that the performance of the EL systems on such datasets is overestimated since it is possible to obtain higher accuracy scores by merely learning the prior. To provide a more adequate evaluation benchmark, we introduce the ShadowLink dataset, which includes 16K short text snippets annotated with entity mentions. We evaluate and report the performance of popular EL systems on the ShadowLink benchmark. The results show a considerable difference in accuracy between more and less common entities for all of the EL systems under evaluation, demonstrating the effect of prior probability bias and entity overshadowing."
}
"""

_DATASET_NAME = "shadowlink"
_DISPLAY_NAME = "SHADOWLINK"

_DESCRIPTION = """\
This dataset contains SHADOWLINK corpus"""

_HOMEPAGE = "https://huggingface.co/datasets/vera-pro/ShadowLink"
_URLs = {
    "top": "https://huggingface.co/datasets/vera-pro/ShadowLink/resolve/main/Top.json",
    "shadow": "https://huggingface.co/datasets/vera-pro/ShadowLink/resolve/main/Shadow.json",
    "tail": "https://huggingface.co/datasets/vera-pro/ShadowLink/resolve/main/Tail.json",
}


logger = datasets.utils.logging.get_logger(__name__)


class SHADOWLINK(GeneratorBasedBuilder):
    """
    SHADOWLINK dataset
    """

    BUILDER_CONFIGS = [
        EntityLinkingDatasetConfig(
            name="top",
            version=VERSION,
            description=_DESCRIPTION,
            subset_id="top",
        ),
        EntityLinkingDatasetConfig(
            name="shadow",
            version=VERSION,
            description=_DESCRIPTION,
            subset_id="shadow",
        ),
        EntityLinkingDatasetConfig(
            name="tail",
            version=VERSION,
            description=_DESCRIPTION,
            subset_id="tail",
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
                name=Split.TEST,
                gen_kwargs={
                    "input_path": dl_manager.download_and_extract(_URLs[self.config.subset_id]),
                    "subset_id": self.config.subset_id,
                },
            ),
        ]

    def _generate_examples(self, input_path: str, subset_id: str) -> Iterator[tuple[str, dict[str, Any]]]:
        """"""
        logger.info(f"Generating examples from {input_path}")
        raw_examples = json.load(open(input_path, 'r', encoding='utf-8'))

        uid = map(str, itertools.count(start=0, step=1))
        for i, r_e in enumerate(raw_examples):
            eid = next(uid)
            example = {
                "dataset": _DATASET_NAME+"_"+subset_id,
                "id": eid,
                "text": r_e["example"],
                "entities": [{
                    "start": r_e["span"][0],
                    "end": r_e["span"][1],
                    "label": [str(r_e["wiki_id"])]
                }],
            }
            yield eid, example

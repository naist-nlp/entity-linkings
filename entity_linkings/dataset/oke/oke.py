"https://www.nzdl.org/wikification/data/wikifiedStories.zip'"

import itertools
from typing import Any, Iterator

import datasets
from datasets import (
    DatasetInfo,
    DownloadManager,
    GeneratorBasedBuilder,
    Split,
    SplitGenerator,
)
from rdflib import Graph, Namespace, URIRef

from ..entity_linkikngs_hub import VERSION, EntityLinkingDatasetConfig, el_features
from ..utils import get_wikipedia_summary, read_text_file

_LOCAL = False
_LANGUAGES = ["English"]
_CITATION = """\
@incollection{NuzzoleseAndreaGiovanni2015OKEC,
    title = {Open Knowledge Extraction Challenge (OKE-2015)},
    author = {Nuzzolese, Andrea Giovanni and Gentile, Anna Lisa and Presutti, Valentina and Gangemi, Aldo and Garigliotti, Darío and Navigli, Roberto and Stankovic, Milan and Cabrio, Elena and Zimmermann, Antoine and Gandon, Fabien},
    address = {Switzerland},
    booktitle = {Semantic Web Evaluation Challenges},
    isbn = {3319255177},
    issn = {1865-0929},
    pages = {3-15},
    publisher = {Springer International Publishing AG},
    series = {Communications in Computer and Information Science},
    volume = {548},
    year = {2015},
}
"""

_DATASET_NAME = "oke"
_DISPLAY_NAME = "OKE"

_DESCRIPTION = """\
This dataset contain OKE-Challenge corpus.
"""

_HOMEPAGE = {
    "2015": "https://github.com/anuzzolese/oke-challenge/",
    "2016": "https://github.com/anuzzolese/oke-challenge-2016"
}
_URLs = {
    "2015": {
        "train": "https://raw.githubusercontent.com/anuzzolese/oke-challenge-2016/refs/heads/master/GoldStandard_sampleData/task1/dataset_task_1.ttl",
        "test": "https://raw.githubusercontent.com/anuzzolese/oke-challenge/refs/heads/master/evaluation-data/task1/evaluation-dataset-task1.ttl"
    },
    "2016": {
        "train": "https://raw.githubusercontent.com/anuzzolese/oke-challenge-2016/refs/heads/master/GoldStandard_sampleData/task1/dataset_task_1.ttl",
        "test": "https://raw.githubusercontent.com/anuzzolese/oke-challenge-2016/refs/heads/master/evaluation-data/task1/evaluation-dataset-task1.ttl"
    }
}
_LICENCE = "unknown"

logger = datasets.utils.logging.get_logger(__name__)


class OKE(GeneratorBasedBuilder):
    """
    OKE-Challenge dataset
    """

    BUILDER_CONFIGS = [
        EntityLinkingDatasetConfig(
            name='2015',
            version=VERSION,
            description=_DESCRIPTION,
            subset_id='2015',
        ),
        EntityLinkingDatasetConfig(
            name='2016',
            version=VERSION,
            description=_DESCRIPTION,
            subset_id='2016',
        ),
    ]

    def _info(self) -> DatasetInfo:
        if self.config.subset_id == '2015':
            homepage = _HOMEPAGE['2015']
        else:
            homepage = _HOMEPAGE['2016']

        return DatasetInfo(
            description=_DESCRIPTION,
            features=el_features,
            homepage=homepage,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: DownloadManager) -> list[SplitGenerator]:
        """Returns SplitGenerators."""
        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={"input_path": dl_manager.download_and_extract(_URLs[self.config.subset_id]['train'])},
            ),

            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={"input_path": dl_manager.download_and_extract(_URLs[self.config.subset_id]['test'])},
            ),
        ]

    @staticmethod
    def get_documents(g: Graph, prefix: Namespace) -> dict[str, str]:
        documents = {}
        for s, p, o in g:
            if str(p) == str(prefix.isString):
                doc_id = str(s).split('#')[0].split('/')[-1]
                documents[str(doc_id)] = str(o)
        return documents

    def _generate_examples(self, input_path: str) -> Iterator[tuple[str, dict[str, Any]]]:
        """"""
        logger.info(f"Generating examples from {input_path}")
        raw_text = read_text_file(input_path)
        prefix = Namespace('http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#')
        g = Graph()
        g.parse(data=raw_text, format='turtle')

        entity_dict = {}
        raw_documents = self.get_documents(g, prefix)

        uid = map(str, itertools.count(start=0, step=1))
        for doc_id, raw_doc in raw_documents.items():
            entities = []
            anchors = [s for s in g.subjects(predicate=prefix.anchorOf) if f'http://www.ontologydesignpatterns.org/data/oke-challenge/task-1/{doc_id}#' in str(s)]
            for anchor in anchors:
                start = g.value(subject=anchor, predicate=prefix.beginIndex)
                end = g.value(subject=anchor, predicate=prefix.endIndex)
                reference = str(g.value(subject=anchor, predicate=URIRef("http://www.w3.org/2005/11/its/rdf#taIdentRef")))
                wikiname = reference.split("/")[-1]
                wikiname = wikiname[len("sentence-"):] if wikiname.startswith("sentence-") else wikiname
                if wikiname not in entity_dict:
                    wiki_summary = get_wikipedia_summary(wikiname)
                    entity_dict[wikiname] = wiki_summary
                pageid = str(entity_dict[wikiname]['pageid'])
                entities.append({"start": start, "end": end, "label": [pageid]})
            example = {"dataset": "oke2015", "id": doc_id, "text": raw_doc, "entities": entities}
            yield next(uid), example

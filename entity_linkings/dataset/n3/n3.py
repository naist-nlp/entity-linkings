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
from rdflib import Graph, Namespace, URIRef

from ..entity_linkikngs_hub import VERSION, EntityLinkingDatasetConfig, el_features
from ..utils import get_wikipedia_summary, read_text_file

_LOCAL = False
_LANGUAGES = ["English"]
_CITATION = """\
@inproceedings{roder-etal-2014-n3,
    title = "N{\textthreesuperior} - A Collection of Datasets for Named Entity Recognition and Disambiguation in the {NLP} Interchange Format",
    author = {R{\"o}der, Michael  and
        Usbeck, Ricardo  and
        Hellmann, Sebastian  and
        Gerber, Daniel  and
        Both, Andreas
    },
    booktitle = "Proceedings of the Ninth International Conference on Language Resources and Evaluation ({LREC}'14)",
    month = may,
    year = "2014",
    address = "Reykjavik, Iceland",
    publisher = "European Language Resources Association (ELRA)",
    url = "https://aclanthology.org/L14-1662/",
    pages = "3529--3533",
}
"""

_DATASET_NAME = "n3-collection"
_DISPLAY_NAME = "N3"

_DESCRIPTION = """\
This dataset contain N3 corpus.
"""

_HOMEPAGE = "https://github.com/dice-group/n3-collection"
_URL = "https://raw.githubusercontent.com/dice-group/n3-collection/refs/heads/master/"
_LICENCE = "unknown"

logger = datasets.utils.logging.get_logger(__name__)


class N3(GeneratorBasedBuilder):
    """
    N3 dataset
    """

    BUILDER_CONFIGS = [
        EntityLinkingDatasetConfig(
            name="r128",
            version=VERSION,
            description=_DESCRIPTION,
            subset_id='r128',
        ),
        EntityLinkingDatasetConfig(
            name="r500",
            version=VERSION,
            description=_DESCRIPTION,
            subset_id='r500',
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
        if self.config.subset_id == 'r128':
            url = os.path.join(_URL, "Reuters-128.ttl")
            print(url)
            data_path = dl_manager.download_and_extract(url)
            return [
                SplitGenerator(
                    name=Split.TEST,
                    gen_kwargs={"data_path": data_path, "subset_id": "r128"},
                ),
            ]
        else:
            url = os.path.join(_URL, "RSS-500.ttl")
            data_path = dl_manager.download_and_extract(url)
            return [
                SplitGenerator(
                    name=Split.TEST,
                    gen_kwargs={"data_path": data_path, "subset_id": "r500"},
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

    def _generate_examples(self, data_path: str, subset_id: str) -> Iterator[tuple[str, dict[str, Any]]]:
        """"""
        logger.info(f"Generating examples from {data_path} for subset: {subset_id}")
        data_name = "Reuters-128" if subset_id == "r128" else "RSS-500"
        raw_text = read_text_file(data_path)
        prefix = Namespace('http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#')
        g = Graph()
        g.parse(data=raw_text, format='turtle')

        entity_dict = {}
        raw_documents = self.get_documents(g, prefix)

        uid = map(str, itertools.count(start=0, step=1))
        for doc_id, raw_doc in raw_documents.items():
            entities = []
            anchors = [s for s in g.subjects(predicate=prefix.anchorOf) if f'http://aksw.org/N3/{data_name}/{doc_id}#' in str(s)]
            for anchor in anchors:
                start = g.value(subject=anchor, predicate=prefix.beginIndex)
                end = g.value(subject=anchor, predicate=prefix.endIndex)
                reference = str(g.value(subject=anchor, predicate=URIRef("http://www.w3.org/2005/11/its/rdf#taIdentRef")))
                if "http://aksw.org/notInWiki" in reference or "http://de.dbpedia.org/" in reference:
                    pageid = "-1"
                else:
                    wikiname = reference.split("/")[-1]
                    if wikiname not in entity_dict:
                        wiki_summary = get_wikipedia_summary(wikiname)
                        entity_dict[wikiname] = wiki_summary
                    pageid = str(entity_dict[wikiname]['pageid'])
                entities.append({"start": start, "end": end, "label": [pageid]})
            example = {"dataset": data_name, "id": doc_id, "text": raw_doc, "entities": entities}
            yield next(uid), example

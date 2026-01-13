import json
import os
from argparse import ArgumentParser

from datasets import DownloadManager
from rdflib import Graph, Namespace, URIRef
from tqdm.auto import tqdm

from appendix.data_convert.utils import WikiMapper, read_text_file

_URLs = {
    "train": "https://raw.githubusercontent.com/anuzzolese/oke-challenge/refs/heads/master/GoldStandard_sampleData/task1/dataset_task_1.ttl",
    "test": "https://raw.githubusercontent.com/anuzzolese/oke-challenge/refs/heads/master/evaluation-data/task1/evaluation-dataset-task1.ttl"
}


def get_documents(g: Graph, prefix: Namespace) -> dict[str, str]:
    documents = {}
    for s, p, o in g:
        if str(p) == str(prefix.isString):
            doc_id = str(s).split('#')[0].split('/')[-1]
            documents[str(doc_id)] = str(o)
    return documents


def main(finder: WikiMapper, output_dir: str) -> None:
    output_dir_data = os.path.join(output_dir, 'oke2015')
    os.makedirs(output_dir_data, exist_ok=True)
    print(f"Output directory data: {output_dir_data}")

    for split, url in _URLs.items():
        data_path = DownloadManager().download_and_extract(url)
        raw_text = read_text_file(data_path)

        prefix = Namespace('http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#')
        g = Graph()
        g.parse(data=raw_text, format='turtle')

        raw_documents = get_documents(g, prefix)
        with open(os.path.join(output_dir_data, f"{split}.jsonl"), 'w', encoding='utf-8') as f:
            pbar = tqdm(total=len(raw_documents), desc="Processing documents")
            for doc_id, raw_doc in raw_documents.items():
                pbar.update()
                entities = []
                anchors = [s for s in g.subjects(predicate=prefix.anchorOf) if f'http://www.ontologydesignpatterns.org/data/oke-challenge/task-1/{doc_id}#' in str(s)]
                for anchor in anchors:
                    start = int(g.value(subject=anchor, predicate=prefix.beginIndex))
                    end = int(g.value(subject=anchor, predicate=prefix.endIndex))
                    reference = str(g.value(subject=anchor, predicate=URIRef("http://www.w3.org/2005/11/its/rdf#taIdentRef")))
                    wikiname = reference.split("/")[-1]
                    wikiname = wikiname[len("sentence-"):] if wikiname.startswith("sentence-") else wikiname
                    res = finder.find_by_title(wikiname)
                    page_id = res["page_id"] if res is not None and res["page_id"] is not None else "-1"
                    entities.append({"start": start, "end": end, "label": [str(page_id)]})
                example = {
                    "subset": "oke2015",
                    "id": doc_id,
                    "text": raw_doc,
                    "entities": entities
                }
                f.write(f"{json.dumps(example, ensure_ascii=False)}\n")
            pbar.close()


if __name__ == "__main__":
    parser = ArgumentParser(description="OKE2015 Entity Disambiguation Script")
    parser.add_argument("--output_dir", "-o", type=str, default='processed_data', help="Path to the output text file")
    parser.add_argument("--wiki_map_file", '-w', type=str, default='wikipedia_to_wikidata.duckdb', help="Path to the wiki mapping file.")
    args = parser.parse_args()

    finder = WikiMapper(args.wiki_map_file)
    main(finder, output_dir=args.output_dir)
    finder.close()

import json
import os
from argparse import ArgumentParser

from datasets import DownloadManager
from rdflib import Graph, Namespace, URIRef

from appendix.data_convert.utils import WikiMapper, read_text_file

_URL = "https://raw.githubusercontent.com/dice-group/n3-collection/refs/heads/master/"



def get_documents(g: Graph, prefix: Namespace) -> dict[str, str]:
    documents = {}
    for s, p, o in g:
        if str(p) == str(prefix.isString):
            doc_id = str(s).split('#')[0].split('/')[-1]
            documents[str(doc_id)] = str(o)
    return documents


def main(finder: WikiMapper, output_dir: str) -> None:
    output_dir_data = os.path.join(output_dir, 'n3')
    os.makedirs(output_dir_data, exist_ok=True)
    print(f"Output directory data: {output_dir_data}")

    for subset_id in ['Reuters-128', 'RSS-500']:
        url = os.path.join(_URL, f"{subset_id}.ttl")
        data_path = DownloadManager().download_and_extract(url)
        raw_text = read_text_file(data_path)

        with open(os.path.join(output_dir_data, f"{subset_id}.jsonl"), 'w', encoding='utf-8') as f:
            prefix = Namespace('http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#')
            g = Graph()
            g.parse(data=raw_text, format='turtle')
            raw_documents = get_documents(g, prefix)

            for doc_id, raw_doc in raw_documents.items():
                entities = []
                anchors = [s for s in g.subjects(predicate=prefix.anchorOf) if f'http://aksw.org/N3/{subset_id}/{doc_id}#' in str(s)]
                for anchor in anchors:
                    start = g.value(subject=anchor, predicate=prefix.beginIndex)
                    end = g.value(subject=anchor, predicate=prefix.endIndex)
                    reference = str(g.value(subject=anchor, predicate=URIRef("http://www.w3.org/2005/11/its/rdf#taIdentRef")))
                    if "http://aksw.org/notInWiki" in reference or "http://de.dbpedia.org/" in reference:
                        page_id = "-1"
                    else:
                        wikiname = reference.split("/")[-1]
                        res = finder.find_by_title(wikiname)
                        page_id = res["page_id"] if res is not None and res["page_id"] is not None else "-1"
                    entities.append({"start": start, "end": end, "label": [str(page_id)]})
                example = {
                    "subset": subset_id.lower(),
                    "id": doc_id,
                    "text": raw_doc,
                    "entities": entities
                }
                f.write(f"{json.dumps(example, ensure_ascii=False)}\n")


if __name__ == "__main__":
    parser = ArgumentParser(description="N3 Entity Disambiguation Script")
    parser.add_argument("--output_dir", "-o", type=str, default='processed_data', help="Path to the output text file")
    parser.add_argument("--wiki_map_file", '-w', type=str, default='wikipedia_to_wikidata.duckdb', help="Path to the wiki mapping file.")
    args = parser.parse_args()

    finder = WikiMapper(args.wiki_map_file)
    main(finder, output_dir=args.output_dir)
    finder.close()

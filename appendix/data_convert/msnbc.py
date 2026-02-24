import json
import os
from argparse import ArgumentParser
from typing import Any
from xml.etree import ElementTree

from datasets import DownloadManager

from appendix.data_convert.utils import WikiMapper

_URL = "https://github.com/dice-group/gerbil/releases/download/v1.2.6/gerbil_data.zip"



def get_documents(data_dir: str) -> dict[str, str]:
    file_names = os.listdir(data_dir)
    documents = {}
    for file_name in file_names:
        doc_id = file_name.split(".")[0]
        with open(os.path.join(data_dir, file_name), 'r', encoding='utf-8') as f:
            documents[doc_id] = f.read()
    return documents


def parse_annotation(file_path: str, finder: WikiMapper) -> list[dict[str, Any]]:
    entities = []
    with open(file_path, 'r', encoding='utf-8') as f:
        g = ElementTree.parse(f)
        root = g.getroot()
        for anno in root.findall('ReferenceInstance'):
            start = anno.findtext('Offset')
            length = anno.findtext('Length')
            wikilink = anno.findtext('ChosenAnnotation')
            surface_form = anno.findtext('SurfaceForm')
            if start is None or length is None or wikilink is None or surface_form is None:
                raise ValueError("Missing required annotation fields.")

            res = finder.find_by_title(wikilink.split('/')[-1].strip())
            page_id = str(res["page_id"]) if res is not None and res["page_id"] is not None else "-1"

            entities.append({
                "start": int(start),
                "end": int(start) + int(length),
                "label": [page_id],
            })
    return entities


def main(finder: WikiMapper, output_dir: str) -> None:
    output_dir_data = os.path.join(output_dir, 'msnbc')
    os.makedirs(output_dir_data, exist_ok=True)
    print(f"Output directory data: {output_dir_data}")

    data_dir = os.path.join(DownloadManager().download_and_extract(_URL), "gerbil_data", "datasets", "MSNBC")
    raw_documents = get_documents(os.path.join(data_dir, "RawTextsSimpleChars_utf8"))

    with open(os.path.join(output_dir_data, "test.jsonl"), 'w', encoding='utf-8') as f:
        for kid, content in raw_documents.items():
            entities = parse_annotation(os.path.join(data_dir, "Problems", f"{kid}.txt"), finder)
            example = {
                "subset": "msnbc",
                "id": kid,
                "text": content,
                "entities": entities
            }
            f.write(f"{json.dumps(example, ensure_ascii=False)}\n")


if __name__ == "__main__":
    parser = ArgumentParser(description="MSNBC Entity Disambiguation Script")
    parser.add_argument("--output_dir", "-o", type=str, default='processed_data', help="Path to the output text file")
    parser.add_argument("--wiki_map_file", '-w', type=str, default='wikipedia_to_wikidata.duckdb', help="Path to the wiki mapping file.")
    args = parser.parse_args()

    finder = WikiMapper(args.wiki_map_file)
    main(finder, output_dir=args.output_dir)
    finder.close()

import json
import os
from argparse import ArgumentParser
from itertools import chain

from datasets import DownloadManager
from tqdm.auto import tqdm

from appendix.data_convert.utils import WikiMapper

_URLs = {
    "train": "https://raw.githubusercontent.com/alteca/Linked-DocRED/refs/heads/main/Linked-Re-DocRED/train_revised.json",
    "validation": "https://raw.githubusercontent.com/alteca/Linked-DocRED/refs/heads/main/Linked-Re-DocRED/dev_revised.json",
    "test": "https://raw.githubusercontent.com/alteca/Linked-DocRED/refs/heads/main/Linked-Re-DocRED/test_revised.json"
}



def _convert(output_dir: str, finder: WikiMapper, source: str) -> None:
    output_dir_data = os.path.join(output_dir, f'docred/{source}')
    os.makedirs(output_dir_data, exist_ok=True)
    print(f"Output directory data: {output_dir_data}")

    for split_name, url in _URLs.items():
        ds = DownloadManager().download_and_extract(url)
        raw_examples = json.load(open(ds, 'r', encoding='utf-8'))

        output_path = os.path.join(output_dir_data, f"{split_name}.jsonl")
        with open(output_path, 'w', encoding='utf-8') as f:
            pbar = tqdm(total=len(raw_examples), desc=f"Processing {split_name}")
            for r_e in raw_examples:
                pbar.update(1)
                sentences = r_e['sents']
                global_sentences = list(chain.from_iterable(sentences))
                entities = []
                for entity in r_e['entities']:
                    entity_info = entity['entity_linking']
                    if entity_info['confidence'] != 'A':
                        continue

                    resource = entity_info[f'{source}_resource']
                    if resource in ['#ignored#', f"#DocRED-{entity['id']}"]:
                        page_id = "-1"
                    else:
                        if source == "wikipedia":
                            res = finder.find_by_title(resource)
                            page_id = str(res["page_id"]) if res is not None and res["page_id"] is not None else "-1"
                        elif source == "wikidata":
                            page_id = resource

                    for mention in entity["mentions"]:
                        sent_id = mention['sent_id']
                        start, end = mention['pos'][0], mention['pos'][1]
                        char_start = len(" ".join(sentences[sent_id][:start])) + (1 if start > 0 else 0)
                        char_end = len(" ".join(sentences[sent_id][:end]))
                        char_context = len(" ".join(list(chain.from_iterable(sentences[: sent_id]))))
                        global_char_start = char_start + char_context + (1 if char_context > 0 else 0)
                        global_char_end = char_end + char_context + (1 if char_context > 0 else 0)
                        entities.append({
                            "start": global_char_start,
                            "end": global_char_end,
                            "label": [page_id],
                        })

                example = {
                    "subset": "linked_re-docred",
                    "id": f"{r_e['title']}",
                    "text": " ".join(global_sentences),
                    "entities": entities,
                }
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
            pbar.close()


def main(finder: WikiMapper, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    _convert(output_dir, finder, source="wikipedia")
    _convert(output_dir, finder, source="wikidata")


if __name__ == "__main__":
    parser = ArgumentParser(description="Linked-DocRed Entity Disambiguation Script")
    parser.add_argument("--output_dir", "-o", type=str, default='processed_data', help="Path to the output text file")
    parser.add_argument("--wiki_map_file", '-w', type=str, default='wikipedia_to_wikidata.duckdb', help="Path to the wiki mapping file.")
    args = parser.parse_args()

    finder = WikiMapper(args.wiki_map_file)
    main(finder, output_dir=args.output_dir)
    finder.close()

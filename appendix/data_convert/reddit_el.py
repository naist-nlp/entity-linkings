import json
import os
from argparse import ArgumentParser
from typing import Any

from tqdm.auto import tqdm

from appendix.data_convert.utils import WikiMapper


def get_documents(input_path: str, id_col: int, text_col: int) -> dict[str, str]:
    documents: dict[str, str] = {}
    post_id = "-1"
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            cols = line.split('\t')
            if len(cols) == 1:
                documents[post_id] += cols[0]
                continue
            doc_id = cols[id_col]
            documents[doc_id] = cols[text_col]
            post_id = doc_id
    return documents


def get_annotation(input_path: str) -> dict[str, Any]:
    annotations: dict[str, Any] = {}

    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        if line.strip():
            cols = line.strip().split('\t')
            doc_id = cols[0]
            mention = cols[2]
            wikiname = cols[3]
            start = int(cols[4])
            end = int(cols[5])
            if doc_id not in annotations:
                annotations[doc_id] = []
            annotations[doc_id].append({
                "mention": mention,
                "start": start,
                "end": end,
                "wikiname": wikiname,
            })
    return annotations


def _convert(
        raw_examples: dict[str, str],
        annotations: dict[str, Any],
        finder: WikiMapper,
        subset: str
    ) -> list[dict[str, Any]]:
    examples = []
    pbar = tqdm(total=len(raw_examples))
    for doc_id, content in raw_examples.items():
        pbar.update()
        entities = []
        content = content.strip()
        for entity in annotations.get(doc_id, []):
            wikiname = entity["wikiname"]
            res = finder.find_by_title(wikiname)
            page_id = res["page_id"] if res is not None and res["page_id"] is not None else "-1"
            assert content[entity["start"]:entity["end"]] == entity["mention"], f"{content[entity['start']:entity['end']]} != {entity['mention']}"

            entities.append({
                "start": entity["start"],
                "end": entity["end"],
                "label": [str(page_id)],
            })

        examples.append({
            "subset": subset,
            "id": doc_id,
            "text": content,
            "entities": entities,
        })
    pbar.close()
    return examples


def convert_posts(input_dir: str, output_dir: str, finder: WikiMapper) -> None:
    raw_examples = get_documents(os.path.join(input_dir, "posts.tsv"), id_col=0, text_col=2)
    annotations = get_annotation(os.path.join(input_dir, "gold_post_annotations.tsv"))
    posts_examples = _convert(raw_examples, annotations, finder, subset="posts")
    with open(os.path.join(output_dir, "posts.jsonl"), 'w', encoding='utf-8') as f:
        for item in posts_examples:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def convert_comments(input_dir: str, output_dir: str, finder: WikiMapper) -> None:
    raw_examples = get_documents(os.path.join(input_dir, "comments.tsv"), id_col=0, text_col=4)
    annotations = get_annotation(os.path.join(input_dir, "gold_comment_annotations.tsv"))
    comments_examples = _convert(raw_examples, annotations, finder, subset="comments")
    with open(os.path.join(output_dir, "comments.jsonl"), 'w', encoding='utf-8') as f:
        for item in comments_examples:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def main(input_dir: str, output_dir: str, finder: WikiMapper) -> None:
    output_dir_data = os.path.join(output_dir, 'reddit')
    os.makedirs(output_dir_data, exist_ok=True)
    print(f"Output directory data: {output_dir_data}")

    convert_posts(input_dir, output_dir_data, finder)
    convert_comments(input_dir, output_dir_data, finder)


if __name__ == "__main__":
    parser = ArgumentParser(description="Reddit Entity Disambiguation Script")
    parser.add_argument("--input_dir", "-i", type=str, required=True, help="Path to the input text file")
    parser.add_argument("--output_dir", "-o", type=str, default='processed_data', help="Path to the output text file")
    parser.add_argument("--wiki_map_file", '-w', type=str, default='wikipedia_to_wikidata.duckdb', help="Path to the wiki mapping file.")
    args = parser.parse_args()

    finder = WikiMapper(args.wiki_map_file)
    main(args.input_dir, args.output_dir, finder)
    finder.close()

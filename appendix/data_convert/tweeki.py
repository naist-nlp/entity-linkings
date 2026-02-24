import ast
import itertools
import json
import os
from argparse import ArgumentParser
from typing import Iterator, Optional

import pandas as pd
from datasets import DownloadManager
from tqdm.auto import tqdm

from appendix.data_convert.utils import WikiMapper

_URLs = {
    "train": [
        "https://github.com/ucinlp/tweeki/raw/refs/heads/main/data/Tweeki_data/Tweeki_data_0.csv.gz",
        "https://github.com/ucinlp/tweeki/raw/refs/heads/main/data/Tweeki_data/Tweeki_data_1.csv.gz",
        "https://github.com/ucinlp/tweeki/raw/refs/heads/main/data/Tweeki_data/Tweeki_data_3.csv.gz",
        "https://github.com/ucinlp/tweeki/raw/refs/heads/main/data/Tweeki_data/Tweeki_data_4.csv.gz",
        "https://github.com/ucinlp/tweeki/raw/refs/heads/main/data/Tweeki_data/Tweeki_data_5.csv.gz",
        "https://github.com/ucinlp/tweeki/raw/refs/heads/main/data/Tweeki_data/Tweeki_data_6.csv.gz",
        "https://github.com/ucinlp/tweeki/raw/refs/heads/main/data/Tweeki_data/Tweeki_data_7.csv.gz",
        "https://github.com/ucinlp/tweeki/raw/refs/heads/main/data/Tweeki_data/Tweeki_data_8.csv.gz",
        "https://github.com/ucinlp/tweeki/raw/refs/heads/main/data/Tweeki_data/Tweeki_data_9.csv.gz",
    ],
    "test": [
        "https://github.com/ucinlp/tweeki/raw/refs/heads/main/data/Tweeki_gold/Tweeki_gold.jsonl"
    ]
}


def _convert(
        words: list[str],
        tags: list[str],
        word_list: list[str|list[str]],
        wiki_list: list[str],
        finder: Optional[WikiMapper] = None
    ) -> tuple[str, list[dict[str, str]]]:
    full_text = ' '.join(words)
    assert len(words) == len(tags) and len(word_list) == len(wiki_list)

    current_char_pos = 0
    token_offsets = []
    for word in words:
        start = current_char_pos
        end = start + len(word)
        token_offsets.append((start, end))
        current_char_pos = end + 1

    token_idx = 0
    entities = []
    for mention, label in zip(word_list, wiki_list):
        mention_token_len = len(mention) if mention else 1
        current_tag = tags[token_idx]

        wiki_id = "-1"
        if label != 0:
            wiki_title, qid = label.split('|')
            if finder is not None:
                res = finder.find_by_title(wiki_title)
                wiki_id = res["page_id"] if res is not None and res["page_id"] is not None else "-1"
            else:
                wiki_id = qid

        if mention and current_tag.startswith('B-') or current_tag.startswith('U-'):
            start_offset = token_offsets[token_idx][0]
            end_offset = token_offsets[token_idx + mention_token_len - 1][1]
            assert full_text[start_offset:end_offset] == ' '.join(words[token_idx: token_idx + mention_token_len])

            entities.append({
                "start": start_offset,
                "end": end_offset,
                # "type": current_tag.split('-')[1],
                "label": [str(wiki_id)]
            })
        token_idx += mention_token_len

    return full_text, entities


def convert_csv_gz(input_path: str, finder: Optional[WikiMapper] = None) -> Iterator[tuple[str, list[dict[str, str]]]]:
    with open(input_path, 'rt', encoding='utf-8') as f:
        df = pd.read_csv(f)[['words', 'tags', 'word_list', 'wiki_list']]
        pbar = tqdm(total=len(df), desc="Processing csv.gz")
        for _, row in df.iterrows():
            pbar.update()
            words = ast.literal_eval(row['words'])
            tags = ast.literal_eval(row['tags'])
            words_list = ast.literal_eval(row['word_list'])
            wiki_list = ast.literal_eval(row['wiki_list'])

            full_text, entities = _convert(words, tags, words_list, wiki_list, finder)
            yield full_text, entities
        pbar.close()


def convert_jsonl(input_path: str, finder: Optional[WikiMapper] = None) -> Iterator[tuple[str, list[dict[str, str]]]]:
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            sentence = data['sentence']
            assert len(data['index']) == len(data['link'])

            entities = []
            for i in range(len(data['index'])):
                assert '|' in data['link'][i] and len(data['link'][i].split('|')) == 2
                title, qid = data['link'][i].split('|')
                title = title.replace(" ", "_")

                page_id = "-1"
                if finder is not None:
                    res = finder.find_by_title(title.strip())
                    page_id = res["page_id"] if res is not None else "-1"

                if page_id == "-1" and finder is not None:
                    res = finder.find_by_wikidata_id(qid.strip())
                    assert len(res) <= 1
                    page_id = res[0]["page_id"] if len(res) == 1 else "-1"

                assert isinstance(page_id, str)
                if page_id == "-1":
                    print(f"Warning: '{title}'({qid}) not found in wiki_map database.")

                entities.append({
                    "start": data['index'][i][0],
                    "end": data['index'][i][1],
                    "label": [str(page_id)]
                })

            yield sentence, entities


def convert(output_dir: str, finder: Optional[WikiMapper] = None) -> None:
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory data: {output_dir}")

    uid = map(str, itertools.count(start=0, step=1))
    with open(os.path.join(output_dir, "train.jsonl"), 'w', encoding='utf-8') as f:
        for url in _URLs['train']:
            data_path = DownloadManager().download_and_extract(url)
            for text, entities in convert_csv_gz(data_path, finder):
                f.write(json.dumps({
                    "subset": "tweeki",
                    "id": next(uid),
                    "text": text,
                    "entities": entities
                }, ensure_ascii=False) + '\n')

    uid = map(str, itertools.count(start=0, step=1))
    with open(os.path.join(output_dir, "test.jsonl"), 'w', encoding='utf-8') as f:
        for url in _URLs['test']:
            data_path = DownloadManager().download_and_extract(url)
            for text, entities in convert_jsonl(data_path, finder):
                f.write(json.dumps({
                    "subset": "tweeki",
                    "id": next(uid),
                    "text": text,
                    "entities": entities
                }, ensure_ascii=False) + '\n')


def main(finder: WikiMapper, output_dir: str) -> None:
    output_dir_data = os.path.join(output_dir, 'tweeki')
    os.makedirs(output_dir_data, exist_ok=True)

    convert(os.path.join(output_dir_data, 'wikidata'))
    convert(os.path.join(output_dir_data, 'wikipedia'), finder)


if __name__ == "__main__":
    parser = ArgumentParser(description="TweeKi Entity Disambiguation Script")
    parser.add_argument("--output_dir", "-o", type=str, default='processed_data', help="Path to the output text file")
    parser.add_argument("--wiki_map_file", '-w', type=str, default='wikipedia_to_wikidata.duckdb', help="Path to the wiki mapping file.")
    args = parser.parse_args()

    finder = WikiMapper(args.wiki_map_file)
    main(finder, output_dir=args.output_dir)
    finder.close()

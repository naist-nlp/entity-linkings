import itertools
import json
import os
from argparse import ArgumentParser
from typing import Any, Iterable, Union

from datasets import DownloadManager

from appendix.data_convert.utils import WikiMapper, _conll_to_example

_URL = "http://resources.mpi-inf.mpg.de/yago-naga/aida/download/KORE50.tar.gz"


def read_conll(
        file: Union[str, bytes, os.PathLike],
        delimiter: str = ' ',
        word_column: int = 0,
        tag_column: int = 1,
        link_column: int = 2
    ) -> Iterable[list[dict[str, Any]]]:
    sentences: list[dict[str, Any]] = []
    words: list[str] = []
    labels: list[str] = []
    links: list[str] = []

    with open(file, mode="r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("-DOCSTART-"):
                if sentences:
                    yield sentences
                    sentences = []
            elif not line:
                if words:
                    sentences.append(_conll_to_example(words, labels, links))
                    words = []
                    labels = []
                    links = []
            else:
                cols = line.split(delimiter)
                if len(cols) == 1:
                    words.append(cols[word_column])
                    labels.append("O")
                    links.append("")
                else:
                    words.append(cols[word_column])
                    labels.append(cols[tag_column])
                    links.append(cols[link_column])
        if words:
            sentences.append(_conll_to_example(words, labels, links))
        if sentences:
            yield sentences


def main(finder: WikiMapper, output_dir: str) -> None:
    output_dir_data = os.path.join(output_dir, 'kore50')
    os.makedirs(output_dir_data, exist_ok=True)
    print(f"Output directory data: {output_dir_data}")

    data_dir = DownloadManager().download_and_extract(_URL)
    data_file = os.path.join(data_dir, "KORE50", "AIDA.tsv")

    with open(os.path.join(output_dir_data, "test.jsonl"), 'w', encoding='utf-8') as f:
        uid = map(str, itertools.count(start=0, step=1))
        for sentences in read_conll(data_file, delimiter='\t', tag_column=1, link_column=3):
            for example in sentences:
                entities = []
                for ent in example['entities']:
                    title  = ent['title'][0]
                    if title == '--NME--':
                        entities.append({"start": ent['start'], "end": ent['end'], "label": ["-1"]})
                        continue
                    res = finder.find_by_title(title)
                    page_id = res["page_id"] if res is not None and res["page_id"] is not None else "-1"
                    entities.append({
                        "start": ent['start'],
                        "end": ent['end'],
                        "label": [str(page_id)],
                    })

                example = {
                    "subset": "kore50",
                    "id": next(uid),
                    "text": example['text'],
                    "entities": entities
                }
                f.write(f"{json.dumps(example, ensure_ascii=False)}\n")


if __name__ == "__main__":
    parser = ArgumentParser(description="KORE50 Entity Disambiguation Script")
    parser.add_argument("--output_dir", "-o", type=str, default='processed_data', help="Path to the output text file")
    parser.add_argument("--wiki_map_file", '-w', type=str, default='wikipedia_to_wikidata.duckdb', help="Path to the wiki mapping file.")
    args = parser.parse_args()

    finder = WikiMapper(args.wiki_map_file)
    main(finder, output_dir=args.output_dir)
    finder.close()

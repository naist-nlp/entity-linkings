import json
import os
from argparse import ArgumentParser
from itertools import accumulate
from typing import Any, Iterable, Union

from appendix.data_convert.utils import _conll_to_example


def read_conll(
        file: Union[str, bytes, os.PathLike],
        delimiter: str = '\t',
    ) -> Iterable[tuple[str, list[dict[str, Any]]]]:
    sentences: list[dict[str, Any]] = []
    words: list[str] = []
    labels: list[str] = []
    links: list[str] = []
    id = ""

    with open(file, mode="r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("-DOCSTART-"):
                if sentences:
                    yield id, sentences
                    sentences = []
                id = line[11:].strip()[1:-1]
            elif not line:
                if words:
                    sentences.append(_conll_to_example(words, labels, links))
                    words = []
                    labels = []
                    links = []
            else:
                cols = line.split(delimiter)
                words.append(cols[0])
                if len(cols) == 1:
                    labels.append('O')
                    links.append('')
                else:
                    labels.append(cols[1])
                    links.append(cols[3] if len(cols) == 4 else cols[5])
    if sentences:
        yield id, sentences


def read_tsv_file(file_path: str) -> dict[str, list[dict]]:
    train, testa, testb = [], [], []
    for id, data in read_conll(file_path, delimiter='\t'):
        texts = [d['text'] for d in data]
        context = " ".join(texts)
        cumsum_lengths = list(accumulate([0] + [len(t) + 1 for t in texts]))
        entities = []
        for i, d in enumerate(data):
            for ent in d['entities']:
                start = ent['start'] + cumsum_lengths[i]
                end = ent['end'] + cumsum_lengths[i]
                title = ["-1"] if ent['title'] == ['--NME--'] else ent['title']
                assert context[start:end] == ent["text"]
                entities.append({"start": start, "end": end, "label": title})
        example = {
            "id": id,
            "text": context,
            "entities": entities,
        }
        if 'testa' in id:
            testa.append(example)
        elif 'testb' in id:
            testb.append(example)
        else:
            train.append(example)
    print(f"Train size: {len(train)}")
    print(f"TestA size: {len(testa)}")
    print(f"TestB size: {len(testb)}")
    return {"train": train, "testa": testa, "testb": testb}

def main(input_file: str, output_dir: str) -> None:
    output_dir_data = os.path.join(output_dir, 'aida')
    os.makedirs(output_dir_data, exist_ok=True)
    print(f"Output directory data: {output_dir_data}")

    data = read_tsv_file(input_file)
    for split in ['train', 'testa', 'testb']:
        with open(os.path.join(output_dir_data, f"{split}.jsonl"), 'w', encoding='utf-8') as f:
            for example in data[split]:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = ArgumentParser(description="KILT Entity Disambiguation Script")
    parser.add_argument("--input_file", "-i", type=str, default="AIDA-YAGO2-dataset.tsv", help="Path to the input text file")
    parser.add_argument("--output_dir", "-o", type=str, default='processed_data', help="Path to the output text file")
    args = parser.parse_args()

    main(args.input_file, args.output_dir)

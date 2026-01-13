

import json
import os
from argparse import ArgumentParser

import requests


def convert_zelda_train(train_file: str, output_dir: str) -> None:
    dataset = []
    with open(train_file) as f:
        for line in f:
            line = json.loads(line)
            assert len(line['index']) == len(line['wikipedia_ids'])
            entities = []
            for span, label in zip(line['index'], line['wikipedia_ids']):
                entities.append({
                    "start": span[0],
                    "end": span[1],
                    "label": [str(label)],
                })
            example = {
                "subset": 'zelda',
                "id": str(line['page_id'])+'_'+str(line['section_name']),
                "text": line["text"],
                "entities": entities
            }
            dataset.append(example)

    num_train = int(len(dataset) * 0.9)
    with open(os.path.join(output_dir, "train.jsonl"), 'w', encoding='utf-8') as f:
        for example in dataset[:num_train]:
            f.write(f"{json.dumps(example, ensure_ascii=False)}\n")
    with open(os.path.join(output_dir, "dev.jsonl"), 'w', encoding='utf-8') as f:
        for example in dataset[num_train:]:
            f.write(f"{json.dumps(example, ensure_ascii=False)}\n")


def convert_zelda_test(output_dir: str) -> None:
    test_dataset_names = ['aida-b', 'cweb', 'wned-wiki', 'reddit-comments', 'reddit-posts', 'tweeki', 'shadowlinks-shadow', 'shadowlinks-tail', 'shadowlinks-top']
    with open(os.path.join(output_dir, "test.jsonl"), 'w', encoding='utf-8') as f:
        for test_dataset_name in test_dataset_names:
            url = f"https://github.com/flairNLP/zelda/raw/refs/heads/main/test_data/jsonl/test_{test_dataset_name}.jsonl"
            urlData = requests.get(url, stream=True)
            for chunk in urlData.iter_lines():
                line = json.loads(chunk)
                assert len(line['index']) == len(line['wikipedia_ids'])
                entities = []
                for span, label in zip(line['index'], line['wikipedia_ids']):
                    entities.append({
                        "start": span[0],
                        "end": span[1],
                        "label": [str(label)],
                    })
                example = {
                    "subset": test_dataset_name,
                    "id": str(line['id']),
                    "text": line["text"],
                    "entities": entities
                }
                f.write(f"{json.dumps(example, ensure_ascii=False)}\n")


def main(train_file: str, output_dir: str) -> None:
    output_dir_data = os.path.join(output_dir, 'zelda')
    os.makedirs(output_dir_data, exist_ok=True)
    print(f"Output directory data: {output_dir_data}")

    convert_zelda_train(train_file, output_dir_data)
    convert_zelda_test(output_dir_data)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train_file", '-t', type=str, default="zelda_train.jsonl", help="Path to the zelda training file.")
    parser.add_argument("--dict_file", '-d', type=str, default="zelda_wiki_dictionary.jsonl", help="Path to the zelda wiki dictionary file.")
    parser.add_argument("--output_dir", '-o', type=str, default="processed_data", help="Output directory for processed data.")
    args = parser.parse_args()

    main(args.train_file, args.output_dir)

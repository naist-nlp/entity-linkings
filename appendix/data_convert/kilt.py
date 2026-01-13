import json
import os
from argparse import ArgumentParser
from pathlib import Path

import requests
from datasets import load_dataset
from tqdm.auto import tqdm

TASKS = ['wned', 'cweb', 'aidayago2']
base_url = "http://dl.fbaipublicfiles.com/KILT"


def _convert(data: dict, subset: str) -> dict:
    mention = data["meta"]["mention"].replace(u'\xa0', ' ')
    left_context = data["meta"]["left_context"].replace(u'\xa0', ' ')
    right_context = data["meta"]["right_context"].replace(u'\xa0', ' ')

    text = f"{left_context} {mention} {right_context}".strip()
    start = len(left_context) + 1 if left_context else 0
    end = start + len(mention)

    assert text[start: end] == mention, (text[start: end], mention)
    if data['output'] != []:
        assert len(data['output']) == 1 and len(data['output'][0]['provenance']) == 1
        wikipedia_id = str(data['output'][0]['provenance'][0]['wikipedia_id'])
    else:
        wikipedia_id = "-1"

    example = {
        "subset": subset,
        "id": data["id"],
        "text": text,
        "entities": [{
            "start": start,
            "end": end,
            "label": [wikipedia_id],
        }],
    }
    return example


def convert_dictionary(output_dir: str) -> None:
    url = os.path.join(base_url, "kilt_knowledgesource.json")
    urlData = requests.get(url, stream=True)
    pbar = tqdm(total=5903530, desc="Downloading dictionary")
    with open(os.path.join(output_dir, "dictionary.jsonl"), 'w', encoding='utf-8') as f:
        for chunk in urlData.iter_lines():
            pbar.update()
            line = json.loads(chunk)
            description = "" if len(line['text']) < 2 else line['text'][1].strip()
            entity = {"id": str(line['wikipedia_id']), "title": line['wikipedia_title'], "text": description}
            f.write(json.dumps(entity, ensure_ascii=False) + "\n")
    pbar.close()
    print(f"Converted dictionary to {output_dir}/dictionary.jsonl")


def convert_to_additional_data(output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for split in ["train", "dev"]:
        pbar = tqdm(total=9000000 if split == 'train' else 10000, desc=f"Downloading {split} additional data")
        url = os.path.join(base_url, f"blink-{split}-kilt.jsonl")
        urlData = requests.get(url, stream=True)
        with open(os.path.join(output_dir, f"{split}.jsonl"), 'w', encoding='utf-8') as f:
            for chunk in urlData.iter_lines():
                pbar.update()
                line = json.loads(chunk)
                example = _convert(line, subset="wikipedia")
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        pbar.close()
    print(f"Converted additional data to {output_dir}")


def convert_dataset(task: str, output_dir: str) -> None:
    kilt_task = load_dataset("facebook/kilt_tasks", task)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for split in kilt_task.keys():
        pbar = tqdm(total=len(kilt_task[split]), desc=f"Converting {task} {split}")
        with open(os.path.join(output_dir, f"{split}.jsonl"), 'w', encoding='utf-8') as f:
            for chunk in kilt_task[split]:
                pbar.update()
                example = _convert(chunk, subset=task)
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        pbar.close()
    print(f"Converted {task} to {output_dir}")


def main(output_dir: str) -> None:
    output_dir_data = os.path.join(output_dir, 'kilt')
    os.makedirs(output_dir_data, exist_ok=True)
    print(f"Output directory data: {output_dir_data}")

    convert_dictionary(output_dir_data)
    convert_to_additional_data(output_dir_data)
    for task in TASKS:
        convert_dataset(task, os.path.join(output_dir_data, task))


if __name__ == "__main__":
    parser = ArgumentParser(description="KILT Entity Disambiguation Script")
    parser.add_argument("--output_dir", "-o", type=str, default='processed_data', help="Path to the output text file")
    args = parser.parse_args()

    main(args.output_dir)

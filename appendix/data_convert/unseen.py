import json
import os
from argparse import ArgumentParser
from typing import Any

from appendix.data_convert.utils import read_jsonl


def _convert(data: dict[str, Any]) -> dict[str, Any]:
    context = " ".join([data["left_context_text"], data["word"]])
    end = len(context)
    start = end - len(data["word"])
    context = " ".join([context, data["right_context_text"]])
    example = {
        "subset": "wikilinks_unseen",
        "id": str(data["docId"]),
        "text": context,
        "entities": [{"start": start, "end": end, "label": [str(data["wikiId"])]}],
    }
    return example


def convert_train(input_dir: str, output_dir: str) -> None:
    with open(os.path.join(output_dir, "train.jsonl"), 'w', encoding='utf-8') as f:
        for i in range(6):
            raw_examples = read_jsonl(os.path.join(input_dir, 'train', f"train_{i}.json"))
            for data in raw_examples:
                example = _convert(data)
                f.write(json.dumps(example, ensure_ascii=False) + '\n')


def convert_dev_test(input_dir: str, output_dir: str) -> list[dict[str, Any]]:
    for split in ["dev", "test"]:
        with open(os.path.join(output_dir, f"{split}.jsonl"), 'w', encoding='utf-8') as f:
            raw_examples = read_jsonl(os.path.join(input_dir, f"{split}.json"))
            examples = []
            for data in raw_examples:
                example = _convert(data)
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        return examples


def main(input_dir: str, output_dir: str) -> None:
    output_dir_data = os.path.join(output_dir, 'wikilinks_unseen')
    os.makedirs(output_dir_data, exist_ok=True)
    print(f"Output directory data: {output_dir_data}")

    convert_train(input_dir, output_dir_data)
    convert_dev_test(input_dir, output_dir_data)


if __name__ == "__main__":
    parser = ArgumentParser(description="WikiLinks Unseen Mentions Script")
    parser.add_argument("--input_dir", "-i", type=str, default='unseen_mentions', help="Path to the input text file")
    parser.add_argument("--output_dir", "-o", type=str, default='processed_data', help="Path to the output text file")
    args = parser.parse_args()

    main(args.input_dir, args.output_dir)

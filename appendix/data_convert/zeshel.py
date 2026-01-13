import json
import os
from argparse import ArgumentParser
from typing import Any

from appendix.data_convert.utils import read_jsonl

DOMAIN_SPLITS = {
    "train": [
        "american_football",
        "doctor_who",
        "fallout",
        "final_fantasy",
        "military",
        "pro_wrestling",
        "starwars",
        "world_of_warcraft",
    ],
    "dev": [
        "coronation_street",
        "muppets",
        "ice_hockey",
        "elder_scrolls",
    ],
    "test": [
        "forgotten_realms",
        "lego",
        "star_trek",
        "yugioh"
    ]
}

def get_documents(input_dir: str, split: str) -> dict[str, Any]:
    documents = {}
    for domain in DOMAIN_SPLITS[split]:
        file_path = os.path.join(input_dir, f"documents/{domain}.json")
        for line in read_jsonl(file_path):
            documents[line["document_id"]] = {"title": line["title"], "text": line["text"], "domain": domain}
    if split == "train":
        assert len(documents) == 332632, f"Expected 332632 documents for train, got {len(documents)}"
    elif split == "dev":
        assert len(documents) == 89549, f"Expected 89549 documents for dev, got {len(documents)}"
    elif split == "test":
        assert len(documents) == 70140, f"Expected 70140 documents for test, got {len(documents)}"
    return documents


def convert_dictionary(raw_datasets: dict[str, Any], output_dir: str) -> None:
    with open(os.path.join(output_dir, "dictionary.jsonl"), 'w', encoding='utf-8') as f:
        for split in raw_datasets:
            for doc_id, doc in raw_datasets[split].items():
                f.write(json.dumps({"id": doc_id, "title": doc["title"], "text": doc["text"], "domain": doc["domain"]}, ensure_ascii=False) + "\n")


def get_mention_dictionary(input_dir: str, split: str) -> dict[str, Any]:
    file_path = os.path.join(input_dir, f"mentions/{split}.json")
    mention_dict: dict[str, Any] = {}
    for line in read_jsonl(file_path):
        context_document_id = line["context_document_id"]
        if context_document_id not in mention_dict:
            mention_dict[context_document_id] = []
        mention_dict[context_document_id].append(line)
    return mention_dict


def convert_dataset(dataset: dict[str, Any], mention_dict: dict[str, Any], output_path: str) -> None:
    mention_count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for doc_id, doc in dataset.items():
            if doc_id not in mention_dict:
                continue
            example = {"subset": doc["domain"], "id": doc_id, "text": doc["text"], "entities": []}
            mentions = mention_dict[doc_id]
            mention_count += len(mentions)
            for mention in mentions:
                start, end, label = mention["start_index"], mention["end_index"], mention["label_document_id"]
                assert ' '.join(doc["text"].split()[start:end+1]) == mention["text"]

                char_start = len(' '.join(doc["text"].split()[:start])) + 1 if start > 0 else 0
                char_end = len(' '.join(doc["text"].split()[:end+1]))
                assert doc["text"][char_start:char_end] == mention["text"]

                example["entities"].append({
                    "start": char_start,
                    "end": char_end,
                    "text": mention["text"],
                    "label": [label],
                    "category": mention["category"]
                })
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    assert mention_count == sum(len(v) for v in mention_dict.values())


def main(input_dir: str, output_dir: str) -> None:
    output_dir_data = os.path.join(output_dir, 'zeshel')
    os.makedirs(output_dir_data, exist_ok=True)
    print(f"Output directory data: {output_dir_data}")

    raw_datasets = {split: get_documents(input_dir, split) for split in ["train", "dev", "test"]}
    convert_dictionary(raw_datasets, output_dir_data)
    for split in ["train", "heldout_train_seen", "heldout_train_unseen"]:
        mention_dict = get_mention_dictionary(input_dir, split)
        if split == "train":
            assert sum(len(v) for v in mention_dict.values()) == 49275
        else:
            assert sum(len(v) for v in mention_dict.values()) == 5000
        convert_dataset(raw_datasets["train"], mention_dict, os.path.join(output_dir_data, f"{split}.jsonl"))

    for split in ["dev", "test"]:
        mention_dict = get_mention_dictionary(input_dir, split if split == "test" else "val")
        assert sum(len(v) for v in mention_dict.values()) == 10000
        convert_dataset(raw_datasets[split], mention_dict, os.path.join(output_dir_data, f"{split}.jsonl"))


if __name__ == "__main__":
    parser = ArgumentParser(description="ZeshEL Script")
    parser.add_argument("--input_dir", "-i", type=str, default='zeshel', help="Path to the input text file")
    parser.add_argument("--output_dir", "-o", type=str, default='processed_data', help="Path to the output text file")
    args = parser.parse_args()

    main(args.input_dir, args.output_dir)

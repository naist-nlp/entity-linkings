import itertools
import json
import os
from argparse import ArgumentParser

from datasets import DownloadManager

_URLs = {
    "top": "https://huggingface.co/datasets/vera-pro/ShadowLink/resolve/main/Top.json",
    "shadow": "https://huggingface.co/datasets/vera-pro/ShadowLink/resolve/main/Shadow.json",
    "tail": "https://huggingface.co/datasets/vera-pro/ShadowLink/resolve/main/Tail.json",
}


def main(output_dir: str) -> None:
    output_dir_data = os.path.join(output_dir, 'shadowlink')
    os.makedirs(output_dir_data, exist_ok=True)
    print(f"Output directory data: {output_dir_data}")

    uid = map(str, itertools.count(start=0, step=1))
    for subset_id, url in _URLs.items():
        data_path = DownloadManager().download_and_extract(url)
        with open(os.path.join(output_dir_data, f"{subset_id}.jsonl"), 'w', encoding='utf-8') as f:
            raw_examples = json.load(open(data_path, 'r', encoding='utf-8'))
            for i, r_e in enumerate(raw_examples):
                eid = next(uid)
                example = {
                    "subset": "shadowlink_"+subset_id,
                    "id": eid,
                    "text": r_e["example"],
                    "entities": [{
                        "start": r_e["span"][0],
                        "end": r_e["span"][0]+r_e["span"][1],
                        "label": [str(r_e["wiki_id"])]
                    }],
                }
                f.write(json.dumps(example, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    parser = ArgumentParser(description="ShadowLink Entity Disambiguation Script")
    parser.add_argument("--output_dir", "-o", type=str, default='processed_data', help="Path to the output text file")
    args = parser.parse_args()

    main(args.output_dir)

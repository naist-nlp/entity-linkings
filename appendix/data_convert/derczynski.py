import itertools
import json
import os
from argparse import ArgumentParser

from tqdm import tqdm

from appendix.data_convert.utils import WikiMapper, read_conll


def main(finder: WikiMapper, input_file: str, output_dir: str) -> None:
    output_dir_data = os.path.join(output_dir, 'derczynski')
    os.makedirs(output_dir_data, exist_ok=True)
    print(f"Output directory data: {output_dir_data}")

    with open(os.path.join(output_dir_data, "test.jsonl"), 'w', encoding='utf-8') as f:
        uid = map(str, itertools.count(start=0, step=1))
        for _, sentences in read_conll(input_file, delimiter='\t', tag_column=2, link_column=1):
            pbar = tqdm(total=len(sentences), desc="Processing sentences")
            for sentence in sentences:
                pbar.update()
                entities = []
                for ent in sentence['entities']:
                    if ent['title'][0] == 'NIL':
                        entities.append({"start": ent['start'], "end": ent['end'], "label": ["-1"]})
                        continue
                    title = ent['title'][0].split('/')[-1]
                    res = finder.find_by_title(title)
                    page_id = res["page_id"] if res is not None and res["page_id"] is not None else "-1"
                    entities.append({
                        "start": ent['start'],
                        "end": ent['end'],
                        "label": [str(page_id)],
                    })
                example = {
                    "subset": "derczynski",
                    "id": next(uid),
                    "text": sentence['text'],
                    "entities": entities,
                }
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
            pbar.close()


if __name__ == "__main__":
    parser = ArgumentParser(description="IPM-NEL2023 Script")
    parser.add_argument("--input_file", "-i", type=str, default="ipm_nel.conll", help="Path to the input text file")
    parser.add_argument("--output_dir", "-o", type=str, default='processed_data', help="Path to the output text file")
    parser.add_argument("--wiki_map_file", '-w', type=str, default='wikipedia_to_wikidata.duckdb', help="Path to the wiki mapping file.")
    args = parser.parse_args()

    finder = WikiMapper(args.wiki_map_file)
    main(finder, args.input_file, output_dir=args.output_dir)
    finder.close()

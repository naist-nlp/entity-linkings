import json
import os
from argparse import ArgumentParser

from datasets import DownloadManager

from appendix.data_convert.utils import WikiMapper, read_text_file

_URL = "https://community.nzdl.org/wikification/data/wikifiedStories.zip"



def main(finder: WikiMapper, output_dir: str) -> None:
    output_dir_data = os.path.join(output_dir, 'aquaint')
    os.makedirs(output_dir_data, exist_ok=True)
    print(f"Output directory data: {output_dir_data}")

    data_dir = os.path.join(DownloadManager().download_and_extract(_URL))
    file_names = [fn for fn in os.listdir(data_dir) if fn.endswith('.htm')]

    with open(os.path.join(output_dir_data, "test.jsonl"), 'w', encoding='utf-8') as f:
        for i, file_name in enumerate(file_names):
            raw_text = read_text_file(os.path.join(data_dir, file_name))
            texts = raw_text.split('<p> ')
            texts[0] = texts[0].split('<h1 id="header">')[1][:-7]
            for i in range(1, len(texts) - 1):
                texts[i] = texts[i][:-7]
            texts[-1] = texts[-1][:-23]

            char_len = 0
            procesed_texts = []
            entities = []
            for text in texts:
                if not text.strip():
                    continue

                current_entity = text.find("[[")
                while current_entity != -1:
                    wikiname = ""
                    surface = ""
                    j = current_entity + 2

                    while text[j] not in ["]", "|"]:
                        wikiname += text[j]
                        j += 1

                    if text[j] == "]":
                        surface = wikiname
                    else:
                        j += 1
                        while text[j] not in ["]", "|"]:
                            surface += text[j]
                            j += 1

                        if text[j] =="|":
                            agreement_score = float(text[j+1: j+4])
                            j += 4
                            if agreement_score < 0.5:
                                text = text[:current_entity] + surface + text[j+2:]
                                current_entity = text.find("[[")
                                continue

                    title = wikiname[0].upper() + wikiname.replace(" ", "_")[1:]
                    res = finder.find_by_title(title)
                    page_id = res["page_id"] if res is not None and res["page_id"] is not None else "-1"
                    entities.append({
                        "start": current_entity + char_len,
                        "end": current_entity + char_len + len(surface),
                        "label": [str(page_id)],
                    })

                    text = text[:current_entity] + surface + text[j+2:]
                    assert surface == text[current_entity: current_entity + len(surface)]
                    current_entity = text.find("[[")
                procesed_texts.append(text)
                char_len += len(text) + 1  # +1 for '\n'

            example = {
                "subset": "aquaint",
                "id": file_name[:-4],
                "text": '\n'.join(procesed_texts),
                "entities": entities,
            }
            f.write(f"{json.dumps(example, ensure_ascii=False)}\n")


if __name__ == "__main__":
    parser = ArgumentParser(description="AQUAINT Entity Disambiguation Script")
    parser.add_argument("--output_dir", "-o", type=str, default='processed_data', help="Path to the output text file")
    parser.add_argument("--wiki_map_file", '-w', type=str, default='wikipedia_to_wikidata.duckdb', help="Path to the wiki mapping file.")
    args = parser.parse_args()

    finder = WikiMapper(args.wiki_map_file)
    main(finder, output_dir=args.output_dir)
    finder.close()

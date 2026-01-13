import html
import json
import os
import re
import unicodedata
from argparse import ArgumentParser
from typing import Any
from xml.etree import ElementTree

from tqdm.auto import tqdm

from appendix.data_convert.utils import WikiMapper


def create_unified_normalization_map(raw_text: str) -> tuple[str, list[int]]:
    intermediate_chars = []
    pattern = re.compile(r"&[a-zA-Z0-9#]+;")
    last_idx = 0

    for match in pattern.finditer(raw_text):
        for i in range(last_idx, match.start()):
            intermediate_chars.append((raw_text[i], i))

        decoded = html.unescape(match.group())
        for char in decoded:
            intermediate_chars.append((char, match.start()))
        last_idx = match.end()

    for i in range(last_idx, len(raw_text)):
        intermediate_chars.append((raw_text[i], i))

    final_text = ""
    old_to_new = [0] * (len(raw_text) + 1)

    current_final_idx = 0
    for char, original_idx in intermediate_chars:
        normalized_char = unicodedata.normalize('NFKC', char)
        final_text += normalized_char

        target_orig_idx = original_idx
        while len(old_to_new) > target_orig_idx and target_orig_idx <= original_idx:
            target_orig_idx += 1

        for i in range(original_idx, len(raw_text) + 1):
            old_to_new[i] = current_final_idx

        current_final_idx += len(normalized_char)

    old_to_new[len(raw_text)] = current_final_idx

    return final_text, old_to_new


def _convert(input_dir: str, finder: WikiMapper, subset: str) -> list[dict[str, Any]]:
    data_dir = os.path.join(input_dir, "wned-datasets", subset)
    files = [fn for fn in os.listdir(os.path.join(data_dir, "RawText"))]
    raw_texts = {}
    index_maps = {}
    pbar = tqdm(total=len(files), desc="Preprocessing raw texts")
    for fn in files:
        pbar.update(1)
        context = open(os.path.join(data_dir, "RawText", fn), 'r', encoding='utf-8').read()
        context, index_map = create_unified_normalization_map(context)
        raw_texts[fn] = context
        index_maps[fn] = index_map
    pbar.close()

    xml_file = os.path.join(data_dir, "clueweb.xml") if subset == 'clueweb12' else os.path.join(data_dir, "wikipedia.xml")

    examples = []
    tree = ElementTree.parse(xml_file)
    root = tree.getroot()

    documents = root.findall("document")
    pbar = tqdm(total=len(documents), desc="Processing documents")
    for elem in documents:
        pbar.update()
        entities = []
        docname = elem.attrib['docName']

        plus_char_num = 0
        for child in elem.findall("annotation"):
            wikiname = child[1].text.replace(' ', '_')
            res = finder.find_by_title(wikiname)
            page_id = res["page_id"] if res is not None and res["page_id"] is not None else "-1"
            pageid = str(page_id)

            start = int(child[2].text)
            end = start + int(child[3].text)
            new_start, new_end = index_maps[docname][start], index_maps[docname][end]
            if child[0].text != raw_texts[docname][new_start:new_end]:
                if docname == 'clueweb12-0503wb-01-33875':
                    if child[0].text == 'Peter Paige' and raw_texts[docname][new_start + plus_char_num: new_end + plus_char_num] == '(Peter Paig':
                        plus_char_num += 1
                    if child[0].text == 'Pittsburgh':
                        if raw_texts[docname][new_start + plus_char_num: new_end + plus_char_num] == ' Pittsburg':
                            plus_char_num += 1
                        if raw_texts[docname][new_start + plus_char_num: new_end + plus_char_num] == 'ittsburrgh.':
                            plus_char_num -= 1
                    if child[0].text == 'QUEER AS FOLK' and raw_texts[docname][new_start + plus_char_num: new_end + plus_char_num] == 's\nQUEER AS FO':
                        plus_char_num += 2
                    new_start += plus_char_num
                    new_end += plus_char_num
            assert child[0].text == raw_texts[docname][new_start:new_end], f"{child[0].text} != {raw_texts[docname][new_start:new_end]}"
            entities.append({"start": new_start, "end": new_end, "label": [pageid]})

        examples.append({
            "subset": subset,
            "id": docname,
            "text": raw_texts[docname],
            "entities": entities,
        })
    pbar.close()
    return examples


def main(input_dir: str, output_dir: str, finder: WikiMapper) -> None:
    output_dir_data = os.path.join(output_dir, 'wned')
    os.makedirs(output_dir_data, exist_ok=True)
    print(f"Output directory data: {output_dir_data}")

    examples = _convert(input_dir, finder, subset="wikipedia")
    with open(os.path.join(output_dir_data, "wikipedia.jsonl"), 'w', encoding='utf-8') as f:
        for item in examples:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    examples = _convert(input_dir, finder, subset="clueweb12")
    with open(os.path.join(output_dir_data, "clueweb.jsonl"), 'w', encoding='utf-8') as f:
        for item in examples:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    parser = ArgumentParser(description="WNED Entity Disambiguation Script")
    parser.add_argument("--input_dir", "-i", type=str, required=True, help="Path to the input text file")
    parser.add_argument("--output_dir", "-o", type=str, default='processed_data', help="Path to the output text file")
    parser.add_argument("--wiki_map_file", '-w', type=str, default='wikipedia_to_wikidata.duckdb', help="Path to the wiki mapping file.")
    args = parser.parse_args()

    finder = WikiMapper(args.wiki_map_file)
    main(args.input_dir, args.output_dir, finder)
    finder.close()

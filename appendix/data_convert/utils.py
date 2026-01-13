import json
import os
from typing import Any, Iterable, Iterator, Union

import duckdb
import requests


def get_wikipedia_summary(title: str, lang: str = 'en') -> dict[str, str]:
    ENDPOINT = f'https://{lang}.wikipedia.org/api/rest_v1/page/summary/'
    HEADERS = {'User-Agent': 'EntityLinkingBenchmark'}

    response = requests.get(ENDPOINT + title, headers=HEADERS)
    if response.status_code == 200:
        data = response.json()
        return {'pageid': str(data.get('pageid', -1)), 'wikibase_id': str(data.get('wikibase_item', -1))}
    else:
        return {'pageid': str(-1), 'wikibase_id': str(-1)}


class WikiMapper:
    def __init__(self, db_path: str) -> None:
        self.con = duckdb.connect(db_path, read_only=True)

    def find_by_title(self, title: str, return_redirect: bool = False) -> dict[str, str] | None:
        # Replace space with underscore to match Wikipedia's internal format
        internal_title = title.replace(" ", "_")

        query = "SELECT page_id, title, wikidata_id, is_redirect, redirect_to FROM wiki_map WHERE title = ?"
        result = self.con.execute(query, [internal_title]).fetchone()

        if result:
            if return_redirect:
                return {
                    "page_id": result[0],
                    "title": result[1],
                    "wikidata_id": result[2],
                    "is_redirect": result[3],
                    "redirect_to": result[4]
                }
            else:
                query_page = "SELECT page_id, title, wikidata_id, is_redirect, redirect_to FROM wiki_map WHERE page_id = ?"
                while True:
                    if not result[3]:
                        break
                    result = self.con.execute(query_page, [result[4]]).fetchone()
                    assert result is not None
                return {"page_id": result[0], "wikidata_id": result[2]}
        return None

    def find_by_page_id(self, page_id: str, return_redirect: bool = False) -> dict[str, str] | None:
        query = "SELECT page_id, title, wikidata_id, is_redirect, redirect_to FROM wiki_map WHERE page_id = ?"
        result = self.con.execute(query, [page_id]).fetchone()

        if result:
            if return_redirect:
                return {
                    "page_id": result[0],
                    "title": result[1],
                    "wikidata_id": result[2],
                    "is_redirect": result[3],
                    "redirect_to": result[4]
                }
            else:
                while True:
                    if not result[3]:
                        break
                    result = self.con.execute(query, [result[4]]).fetchone()
                    assert result is not None
                return {"page_id": result[0], "wikidata_id": result[2]}
        return None

    def find_by_wikidata_id(self, wikidata_id: str) -> list[dict[str, str]]:
        query = "SELECT page_id, title, wikidata_id, is_redirect, redirect_to FROM wiki_map WHERE wikidata_id = ?"
        results = self.con.execute(query, [wikidata_id]).fetchall()

        return [
            {
                "page_id": result[0],
                "title": result[1],
                "wikidata_id": result[2],
                "is_redirect": result[3],
                "redirect_to": result[4],
            } for result in results
        ]

    def close(self) -> None:
        self.con.close()


def read_conll(
        file: Union[str, bytes, os.PathLike],
        delimiter: str = ' ',
        word_column: int = 0,
        tag_column: int = 1,
        link_column: int = 2
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
            elif line.startswith("# "):
                id = line[2:].strip().split('\t')[0]
            elif not line:
                if words:
                    sentences.append(_conll_to_example(words, labels, links))
                    words = []
                    labels = []
                    links = []
            else:
                cols = line.split(delimiter)
                words.append(cols[word_column])
                labels.append(cols[tag_column])
                links.append(cols[link_column])

    if sentences:
        yield id, sentences


def _conll_to_example(words: list[str], tags: list[str], links: list[str]) -> dict[str, Any]:
    text, positions = _conll_words_to_text(words)
    entities = [
        {"start": positions[start][0], "end": positions[end - 1][1], "label": [label], "title": [title], 'text': text[positions[start][0]: positions[end - 1][1]]}
        for start, end, label, title in _conll_tags_to_spans(tags, links)
    ]
    return {"text": text, "entities": entities}


def _conll_words_to_text(words: Iterable[str]) -> tuple[str, list[tuple[int, int]]]:
    text = ""
    positions = []
    offset = 0
    for word in words:
        if text:
            text += " "
            offset += 1
        text += word
        n = len(word)
        positions.append((offset, offset + n))
        offset += n
    return text, positions


def _conll_tags_to_spans(tags: Iterable[str], links: Iterable[str]) -> Iterable[tuple[int, int, str, str]]:
    # NOTE: assume BIO scheme
    start, label, link = -1, None, None
    for i, (tag, link_tag) in enumerate(zip(list(tags) + ["O"], list(links) + ["O"])):
        if tag == "O":
            if start >= 0:
                assert label is not None and link is not None
                yield (start, i, label, link)
                start, label, link = -1, None, None
        else:
            cur_label = tag[2:]
            cur_link = link_tag[2:] if tag[:2] == link_tag[:2] else link_tag
            if tag.startswith("B"):
                if start >= 0:
                    assert label is not None and link is not None
                    yield (start, i, label, link)
                start, label, link = i, cur_label, cur_link
            else:
                if cur_label != label:
                    if start >= 0:
                        assert label is not None and link is not None
                        yield (start, i, label, link)
                    start, label, link = i, cur_label, cur_link


def read_text_file(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


def read_jsonl(file_path: str) -> Iterator[dict[str, Any]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)

import json
import os
from typing import Any, Callable, Iterable, Iterator, Optional, Union

import requests
from datasets import Dataset
from transformers import TrainingArguments


def preprocess(
        dataset: Dataset,
        processing_func: Callable[[Dataset], Iterator[Any]],
        training_arguments: Optional[TrainingArguments]=None
        ) -> dict[str, Dataset]:
    def _preprocess(documents: Dataset) -> Any:
        features = [_ for _ in processing_func(documents)]
        outputs = {}
        for k in list(features[0].keys()):
            outputs[k] = [f[k] for f in features]
        return outputs

    if training_arguments:
        with training_arguments.main_process_first(desc="dataset map pre-processing"):
            column_names = dataset.column_names
            splits = dataset.map(_preprocess, batched=True, remove_columns=column_names)
    else:
        column_names = dataset.column_names
        splits = dataset.map(_preprocess, batched=True, remove_columns=column_names)

    return splits


def get_wikipedia_summary(title: str, lang: str = 'en') -> dict[str, int]:
    ENDPOINT = f'https://{lang}.wikipedia.org/api/rest_v1/page/summary/'
    HEADERS = {'User-Agent': 'EntityLinkingBenchmark'}

    response = requests.get(ENDPOINT + title, headers=HEADERS)
    if response.status_code == 200:
        data = response.json()
        return {'pageid': data.get('pageid', -1), 'wikibase_id': data.get('wikibase_item', -1)}
    else:
        return {'pageid': -1, 'wikibase_id': -1}


def read_text_file(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def read_jsonl(file_path: str) -> Iterator[dict[str, Any]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


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

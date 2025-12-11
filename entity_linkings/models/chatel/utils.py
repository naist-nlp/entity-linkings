import hashlib
import json
import os
import re
from typing import Any

import httpx

DEFAULT_TIMEOUT = httpx.Timeout(150.0, read=150.0, write=50.0, connect=6.0)


def process_multi_choice_prompt(multi_choice_prompt_result: str, candidates: list[str]) -> int:
    L = len(candidates)
    if L == 0:
        return -1
    elif L == 1:
        return 0
    elif 'None of the entity match' in multi_choice_prompt_result:
        return -1

    # update the index finding schema with regular expression.
    index_list = [int(s) - 1 for s in re.findall(r'\b\d+\b', multi_choice_prompt_result) if 0 <= int(s) - 1 < len(candidates)]

    # consider direct index answer of chatgpt
    if len(index_list) == 1:
        return index_list[0]

    # if there are two choices and candidate entities length is more than 2, select the first one.
    if len(index_list) == 2 and len(candidates) > 2:
        return index_list[0]

    # consider complete string match
    index_list = []
    for index, candidate in enumerate(candidates):
        if candidate.lower() in multi_choice_prompt_result.lower():
            add_flag = True
            other_candidates = candidates[:index] + candidates[index+1:]
            for other_candidate in other_candidates:
                if candidate in other_candidate:
                    add_flag = False
                    break
            if add_flag:
                index_list.append(index)

    if len(index_list) == 1:
        return index_list[0]
    return -1


class Cache:
    def __init__(self, model_name: str, cache_dir: str = '.cache') -> None:
        # Save GPT output
        os.makedirs(cache_dir, exist_ok = True)
        self.file_name_cache = f'{cache_dir}/{model_name}.jsonl'
        if os.path.exists(self.file_name_cache):
            # This file contains hash values and raw data
            self.cache = self.load_json()
        else:
            self.cache = dict()

    def serialize(self, obj: Any) -> Any:
        """ Serialize an object to a JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self.serialize(v) for k, v in obj.items()}
        elif hasattr(obj, "__dict__"):
            return self.serialize(obj.__dict__)
        elif isinstance(obj, list):
            return [self.serialize(v) for v in obj]
        else:
            return obj

    @staticmethod
    def create_hash(prompt: str) -> str:
        # Save on an instruction basis
        return hashlib.md5(prompt.encode()).hexdigest()

    def load_json(self) -> dict[str, Any]:
        data = dict()
        with open(self.file_name_cache) as f:
            json_lines = f.readlines()
            for line in json_lines:
                json_obj = json.loads(line)
                data[json_obj['id']] = json_obj['results']
        return data

    def append_to_jsonl(self, instruction: str, results: dict[str, Any]) -> None:
        # Save data as cache.
        self.cache[self.create_hash(instruction)] = results
        save_data = {'id': self.create_hash(instruction), 'results': results}
        with open(self.file_name_cache, 'a') as file:
            json_str = json.dumps(save_data, ensure_ascii=False)
            file.write(json_str + '\n')

    def __call__(self, item: str) -> dict[str, Any]:
        return self.cache[self.create_hash(item)]

    def check_in_cache(self, instruction: str) -> bool:
        return True if self.create_hash(instruction) in self.cache else False

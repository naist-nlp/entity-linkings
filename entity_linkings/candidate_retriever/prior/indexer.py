import json
import os
import pickle
import random
import re
from collections.abc import Callable
from logging import getLogger
from typing import Optional

import numpy as np
from tqdm.auto import tqdm

from entity_linkings.data_utils import EntityDictionary

from ..base import IndexerBase

logger = getLogger(__name__)
logger.setLevel("INFO")

punc_remover = re.compile(r"[\W]+")


def _merge_by_simplified(
        mention_entities_counter: dict[str, dict[str, int]],
        simplify: Callable[[str], str],
    ) -> dict[str, dict[str, int]]:
    '''Group mentions whose simplified form collides, summing their entity counts.

    Each bucket starts as a fresh dict. Storing the caller's dict here instead would
    alias it, and the next mention that simplifies to the same key would add its counts
    straight into mention_id_counter, which is the exact-match table.
    '''
    merged: dict[str, dict[str, int]] = {}
    for mention, entity_counts in mention_entities_counter.items():
        bucket = merged.setdefault(simplify(mention), {})
        for entity, count in entity_counts.items():
            bucket[entity] = bucket.get(entity, 0) + count
    return merged


def build_simpler_mentions_dict(mention_entities_counter: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    # mention without blanks and lower cased
    return _merge_by_simplified(mention_entities_counter, lambda mention: mention.replace(' ', '').lower())


def build_most_simpler_mentions_dict(mention_entities_counter: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    return _merge_by_simplified(mention_entities_counter, lambda mention: punc_remover.sub("", mention.lower()))


class MentionPriorIndexer(IndexerBase):
    def __init__(self, dictionary: EntityDictionary, mention_counter_path: Optional[str] = None) -> None:
        super().__init__(dictionary)
        self.mention_counter_path = mention_counter_path

    def build_index(self, index_path: str | None = None) -> None:
        if index_path is not None and os.path.exists(os.path.join(index_path, "mention_entities_counter.pickle")):
            logger.info(f"Loading existing index from {index_path}")
            self.load(index_path=index_path)
        else:
            if self.mention_counter_path is None:
                raise ValueError("mention_counter_path must be provided to build the index.")
            counter = json.load(open(self.mention_counter_path, 'r'))
            self.meta_ids_to_keys = {entity["name"]: entity["id"] for entity in self.dictionary}
            self.mention_id_counter: dict[str, dict[str, int]] = {}
            pbar = tqdm(total=len(counter), desc='Building index')
            for mention, entity_counter in counter.items():
                pbar.update()
                mention_counter = {}
                for entity_name, count in entity_counter.items():
                    entity_name = entity_name.replace(' ', '_')
                    if entity_name in self.meta_ids_to_keys:
                        wiki_id = self.dictionary.get_from_title(entity_name)['id']
                        mention_counter[wiki_id] = count
                self.mention_id_counter[mention] = mention_counter
            pbar.close()
            self.simpler_mentions_candidate_dict = build_simpler_mentions_dict(self.mention_id_counter)
            self.even_more_simpler_mentions_candidate_dict = build_most_simpler_mentions_dict(self.mention_id_counter)

    def search_knn(self, query: str|list[str], top_k: int, ignore_ids: Optional[list[str]|list[list[str]]] = None) -> tuple[np.ndarray, list[list[str]]]:
        if top_k <= 0:
            raise RuntimeError("K is zero or under zero.")

        queries = [query] if type(query) is str else query
        scores, indices_keys = [], []

        unwanted = self.normalize_ignore_ids(ignore_ids, len(queries))
        max_ignored = max((len(ids) for ids in unwanted), default=0)

        # Short rows cannot be returned, because every row of the score array has to be
        # the same length, so cap K by what the dictionary can still supply for the most
        # heavily filtered query. Without this the padding loop below would be asked for
        # more entities than are left and would never finish.
        selectable = self.num_entities - max_ignored
        if top_k > selectable:
            logger.warning(f"K is over the number of selectable entities. K is modified to {selectable}.")
            top_k = selectable

        for i, query in enumerate(queries):
            current_ignore_set = set(unwanted[i])

            candidates = self.mention_id_counter.get(query, None)
            if not candidates:
                candidates = self.simpler_mentions_candidate_dict.get(query.lower().replace(' ', ''), None)
            if not candidates:
                candidates = self.even_more_simpler_mentions_candidate_dict.get(punc_remover.sub("", query.lower()), None)

            final_ids, final_scores = [], []
            if candidates:
                valid_candidate_ids = [(k, v) for k, v in candidates.items() if k not in current_ignore_set]
                topk_items = sorted(valid_candidate_ids, key=lambda x: x[1], reverse=True)[:top_k]
                if topk_items:
                    num_mentions = sum([x[1] for x in topk_items])
                    final_ids = [x[0] for x in topk_items]
                    final_scores = [x[1] / num_mentions for x in topk_items]
                    current_ignore_set.update(final_ids)

            num_needed = top_k - len(final_ids)
            if num_needed > 0:
                if not candidates:
                    logger.info(f'No candidates found for mention "{query}".')
                # One sample covers the usual case, where the ignored ids are a tiny
                # fraction of the dictionary. Walking the ids is the fallback for when
                # too many of the sampled ones were ignored; it is bounded, whereas
                # sampling until the quota is met is not.
                sample_size = min(num_needed * 2 + 50, self.num_entities)
                fillers: list[str] = []
                for pool in (random.sample(self.entity_ids, sample_size), self.entity_ids):
                    for entity_id in pool:
                        if entity_id in current_ignore_set:
                            continue
                        fillers.append(entity_id)
                        current_ignore_set.add(entity_id)
                        if len(fillers) == num_needed:
                            break
                    if len(fillers) == num_needed:
                        break
                final_ids.extend(fillers)
                final_scores.extend([0.0] * len(fillers))
            indices_keys.append(final_ids)
            scores.append(final_scores)

        return np.array(scores), indices_keys

    def load(self, index_path: str) -> None:
        logger.info("Deserializing index from %s", index_path)
        mention_entities_counter_file = os.path.join(index_path, "mention_entities_counter.pickle")
        simpler_mentions_candidate_file = os.path.join(index_path, "simpler_mentions_candidate_dict.pickle")
        even_more_simpler_mentions_candidate_file = os.path.join(index_path, "even_more_simpler_mentions_candidate_dict.pickle")
        meta_file = os.path.join(index_path, "meta_candidate_list.json")
        self.mention_id_counter = pickle.load(open(mention_entities_counter_file, 'rb'))
        self.simpler_mentions_candidate_dict = pickle.load(open(simpler_mentions_candidate_file, 'rb'))
        self.even_more_simpler_mentions_candidate_dict = pickle.load(open(even_more_simpler_mentions_candidate_file, 'rb'))
        self.meta_ids_to_keys = json.load(open(meta_file))
        logger.info(
            "Loaded index of type %s and size %d", type(self.mention_id_counter), len(self.meta_ids_to_keys)
        )

    def save_index(self, index_path: str, ensure_ascii: bool = False) -> None:
        if not os.path.isdir(index_path):
            os.makedirs(index_path, exist_ok=True)
        logger.info("Serializing index to %s", index_path)
        mention_entities_counter_file = os.path.join(index_path, "mention_entities_counter.pickle")
        simpler_mentions_candidate_file = os.path.join(index_path, "simpler_mentions_candidate_dict.pickle")
        even_more_simpler_mentions_candidate_file = os.path.join(index_path, "even_more_simpler_mentions_candidate_dict.pickle")
        meta_file = os.path.join(index_path, "meta_candidate_list.json")
        pickle.dump(self.mention_id_counter, open(mention_entities_counter_file, 'wb'))
        pickle.dump(self.simpler_mentions_candidate_dict, open(simpler_mentions_candidate_file, 'wb'))
        pickle.dump(self.even_more_simpler_mentions_candidate_dict, open(even_more_simpler_mentions_candidate_file, 'wb'))
        json.dump(self.meta_ids_to_keys, open(meta_file, 'w'), ensure_ascii=ensure_ascii)

    def __len__(self) -> int:
        return len(self.dictionary)

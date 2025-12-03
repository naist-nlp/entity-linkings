import json
import os
import pickle
import random
import re
from dataclasses import dataclass
from logging import getLogger
from typing import Optional

import numpy as np
from datasets import DownloadManager

from entity_linkings.entity_dictionary import EntityDictionaryBase

from ..base import IndexerBase

logger = getLogger(__name__)

punc_remover = re.compile(r"[\W]+")

PATH_TO_REPOSITORY = "https://nlp.informatik.hu-berlin.de/resources/datasets/zelda/zelda_full.zip"


def build_other_dictionaries(mention_entities_counter: dict[str, dict[str, int]]) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    # to improve the recall of the candidate lists we add a lower cased and a further reduced version of each mention to the mention set
    simpler_mentions_candidate_dict: dict[str, dict[str, int]] = {}
    for mention in mention_entities_counter:
        # create mention without blanks and lower cased
        simplified_mention = mention.replace(' ', '').lower()
        # the simplified mention already occurred from another mention
        if simplified_mention in simpler_mentions_candidate_dict:
            for entity in mention_entities_counter[mention]:
                if entity in simpler_mentions_candidate_dict[simplified_mention]:
                    simpler_mentions_candidate_dict[simplified_mention][entity] += mention_entities_counter[mention][entity]
                else:
                    simpler_mentions_candidate_dict[simplified_mention][entity] = mention_entities_counter[mention][entity]
        # its the first occurrence of the simplified mention
        else:
            simpler_mentions_candidate_dict[simplified_mention] = mention_entities_counter[mention]

    even_more_simpler_mentions_candidate_dict: dict[str, dict[str, int]] = {}
    for mention in mention_entities_counter:
        # create simplified mention
        simplified_mention=punc_remover.sub("", mention.lower())
        # the simplified mention already occurred from another mention
        if simplified_mention in even_more_simpler_mentions_candidate_dict:
            for entity in mention_entities_counter[mention]:
                if entity in even_more_simpler_mentions_candidate_dict[simplified_mention]:
                    even_more_simpler_mentions_candidate_dict[simplified_mention][entity] += mention_entities_counter[mention][entity]
                else:
                    even_more_simpler_mentions_candidate_dict[simplified_mention][entity] = mention_entities_counter[mention][entity]
        # its the first occurrence of the simplified mention
        else:
            even_more_simpler_mentions_candidate_dict[simplified_mention] = mention_entities_counter[mention]
    return simpler_mentions_candidate_dict, even_more_simpler_mentions_candidate_dict


class ZeldaCandidateIndexer(IndexerBase):
    @dataclass
    class Config(IndexerBase.Config): ...

    def __init__(self, dictionary: EntityDictionaryBase, config: Optional[Config] = None) -> None:
        super().__init__(dictionary, config)

    def _initialize(self) -> None:
        pass

    def build_index(self, index_path: Optional[str] = None) -> None:
        if index_path and os.path.exists(index_path):
            self.load(index_path)
        else:
            data_dir = os.path.join(DownloadManager().download_and_extract(PATH_TO_REPOSITORY), "zelda")
            with open(os.path.join(data_dir, 'other', 'zelda_mention_entities_counter.pickle'), 'rb') as handle:
                self.mention_entities_counter = pickle.load(handle)
                self.meta_ids_to_keys = {entity["name"]: entity["id"] for entity in self.dictionary}
                self.simpler_mentions_candidate_dict, self.even_more_simpler_mentions_candidate_dict = build_other_dictionaries(self.mention_entities_counter)

    def search_knn(self, query: str|list[str], top_k: int, ignore_ids: Optional[list[str]|list[list[str]]] = None) -> tuple[np.ndarray, list[list[str]]]:
        if top_k <= 0:
            raise RuntimeError("K is zero or under zero.")
        if top_k > len(self.dictionary):
            top_k = len(self.dictionary)
            logger.warning(f"K is over size of dictionary. K is modified to size of dictionary to {len(self.dictionary)}")

        additional_top_k = 0
        if ignore_ids is not None:
            if isinstance(ignore_ids[0], list):
                additional_top_k = max([len(ids) for ids in ignore_ids])
            else:
                additional_top_k = len(ignore_ids)

        queries = [query] if type(query) is str else query
        entity_names = self.dictionary.get_entity_names()
        scores, indices_keys = [], []
        for i, query in enumerate(queries):
            ignore_titles = [self.dictionary(iid)['name'].replace('_', ' ') for iid in ignore_ids[i]] if ignore_ids else []
            candidates = self.mention_entities_counter.get(query, None)
            if not candidates:
                candidates = self.simpler_mentions_candidate_dict.get(query.lower().replace(' ', ''), None)
            if not candidates:
                candidates = self.even_more_simpler_mentions_candidate_dict.get(punc_remover.sub("", query.lower()), None)

            if not candidates:
                logger.info(f'No candidates found for mention "{query}".')
                scores.append([0.]*(top_k))
                random_samples = random.sample(range(len(self.dictionary)), top_k+additional_top_k)
                indices_keys.append([
                    self.dictionary[r_s]['id'] for r_s in random_samples
                    if entity_names[r_s] not in ignore_titles][:top_k]
                )
            else:
                candidate_ids = {}
                for k, v in candidates.items():
                    wiki_title = self.dictionary.get_from_title(k.replace(' ', '_'))['name']
                    wiki_id = self.dictionary.get_from_title(k.replace(' ', '_'))['id']
                    if k not in ignore_titles and wiki_title not in ignore_titles:
                        if wiki_id not in candidate_ids :
                            candidate_ids[wiki_id] = 0
                        candidate_ids[wiki_id] += v

                num_mentions = sum(candidate_ids.values())
                candidate_scores = [c_count / num_mentions for c_count in candidate_ids.values()]

                if len(candidate_ids) < top_k:
                    scores.append(candidate_scores + [0.] * (top_k - len(candidate_ids)))
                    random_samples = random.sample(range(len(entity_names)), top_k + additional_top_k)
                    random_candidates = []
                    for r_s in random_samples:
                        dic_id = self.dictionary[r_s]['id']
                        if entity_names[r_s] not in ignore_titles and dic_id not in candidate_ids:
                            random_candidates.append(dic_id)
                    indices_keys.append(list(candidate_ids.keys())+random_candidates[:top_k - len(candidate_ids)])
                else:
                    indices_keys.append(list(candidate_ids.keys())[:top_k])
                    scores.append(candidate_scores[:top_k])
        return np.array(scores), indices_keys

    def load(self, index_path: str) -> None:
        logger.info("Deserializing index from %s", index_path)
        mention_entities_counter_file = os.path.join(index_path, "zelda_mention_entities_counter.pickle")
        simpler_mentions_candidate_file = os.path.join(index_path, "zelda_simpler_mentions_candidate_dict.pickle")
        even_more_simpler_mentions_candidate_file = os.path.join(index_path, "zelda_even_more_simpler_mentions_candidate_dict.pickle")
        meta_file = os.path.join(index_path, "meta_zelda_candidate_list.json")
        self.mention_entities_counter = pickle.load(open(mention_entities_counter_file, 'rb'))
        self.simpler_mentions_candidate_dict = pickle.load(open(simpler_mentions_candidate_file, 'rb'))
        self.even_more_simpler_mentions_candidate_dict = pickle.load(open(even_more_simpler_mentions_candidate_file, 'rb'))
        self.meta_ids_to_keys = json.load(open(meta_file))
        logger.info(
            "Loaded index of type %s and size %d", type(self.mention_entities_counter), len(self.meta_ids_to_keys)
        )

    def save_index(self, index_path: str, ensure_ascii: bool = False) -> None:
        if not os.path.isdir(index_path):
            os.mkdir(index_path)
        logger.info("Serializing index to %s", index_path)
        mention_entities_counter_file = os.path.join(index_path, "zelda_mention_entities_counter.pickle")
        simpler_mentions_candidate_file = os.path.join(index_path, "zelda_simpler_mentions_candidate_dict.pickle")
        even_more_simpler_mentions_candidate_file = os.path.join(index_path, "zelda_even_more_simpler_mentions_candidate_dict.pickle")
        meta_file = os.path.join(index_path, "meta_zelda_candidate_list.json")
        pickle.dump(self.mention_entities_counter, open(simpler_mentions_candidate_file, 'wb'))
        pickle.dump(self.simpler_mentions_candidate_dict, open(even_more_simpler_mentions_candidate_file, 'wb'))
        pickle.dump(self.even_more_simpler_mentions_candidate_dict, open(mention_entities_counter_file, 'wb'))
        json.dump(self.meta_ids_to_keys, open(meta_file, 'w'), ensure_ascii=ensure_ascii)

    def __len__(self) -> int:
        return len(self.dictionary)

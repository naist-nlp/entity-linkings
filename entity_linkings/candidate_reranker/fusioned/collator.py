import logging
import random
from dataclasses import dataclass
from typing import Any, Optional, Union

from transformers import BatchEncoding
from transformers.data.data_collator import pad_without_fast_tokenizer_warning

from entity_linkings.data_utils import CollatorBase, EntityDictionary

logger = logging.getLogger(__name__)



@dataclass
class CollatorForFusioned(CollatorBase):
    dictionary: Optional[EntityDictionary] = None
    train: bool = False

    def __call__(self, features: list[Union[dict[str, Any], BatchEncoding]]) -> BatchEncoding:
        if self.dictionary is None:
            raise ValueError("this modules needs to dictionary.")

        features = [f.copy() for f in features]
        new_features, batch_labels = [], []
        for f in features:
            context_tokens = f.pop("input_ids")
            candidates_ids = f.pop("candidates")
            candidates = [self.dictionary(cand) for cand in candidates_ids]

            golds = f.pop("labels", None)
            if golds is not None:
                gold_entity = self.dictionary(golds[0])
                batch_labels.append(gold_entity['name'])
                if self.train:
                    candidates = [gold_entity] + candidates[:-1]
                    random.shuffle(candidates)

            for cand in candidates:
                encoding = self.tokenizer.prepare_for_model(
                    context_tokens + cand['encoding'],
                    truncation=True,
                    add_special_tokens=True,
                )
                new_features.append(encoding)

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            new_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        for k, v in batch.items():
            cnum = len(v) // len(features)
            v = v.view(len(features), cnum, -1)
            batch[k] = v if self.return_tensors == "pt" else v.tolist()

        if batch_labels:
            batch['labels'] = self.tokenizer(
                batch_labels,
                padding=True,
                truncation=True,
                return_tensors=self.return_tensors,
            ).input_ids
        return batch

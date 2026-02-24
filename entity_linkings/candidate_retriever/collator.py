import logging
import random
from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
from transformers import BatchEncoding
from transformers.data.data_collator import pad_without_fast_tokenizer_warning

from entity_linkings.data_utils import CollatorBase, EntityDictionary

logger = logging.getLogger(__name__)


@dataclass
class CollatorForRetrieval(CollatorBase):
    dictionary: Optional[EntityDictionary] = None
    num_hard_negatives: int = 0
    random_negative_sampling: bool = False

    def __call__(self, features: list[Union[dict[str, Any], BatchEncoding]]) -> BatchEncoding:
        if self.dictionary is None:
            raise ValueError("this modules needs to dictionary.")

        features = [f.copy() for f in features]
        inbatch_ids, hard_negative_ids = [], []
        for f in features:
            labels = f.pop("labels")
            inbatch_ids.append(labels[0])
            candidates = f.pop("candidates", None)
            if candidates is not None and self.num_hard_negatives > 0:
                if self.random_negative_sampling:
                    hard_negative_ids.extend(random.sample(candidates, self.num_hard_negatives))
                else:
                    hard_negative_ids.extend(candidates[:self.num_hard_negatives])

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        candidates_labels = list(set(inbatch_ids + list(set(hard_negative_ids))))
        candidates_encodings = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            [self.dictionary(cid)['encoding'] for cid in candidates_labels],
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch.update({f'candidates_{k}': v for k, v in candidates_encodings.items()})

        candidates_idx = {cid: idx for idx, cid in enumerate(candidates_labels)}
        inbatch_labels = [candidates_idx[cid] for cid in inbatch_ids]
        batch["labels"] = torch.tensor(inbatch_labels) if self.return_tensors == "pt" else inbatch_labels

        return batch

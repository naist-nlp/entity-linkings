from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
from transformers import BatchEncoding
from transformers.data.data_collator import pad_without_fast_tokenizer_warning

from entity_linkings.data_utils import (
    CollatorBase,
    EntityDictionary,
)


@dataclass
class CollatorForCrossEncoder(CollatorBase):
    dictionary: Optional[EntityDictionary] = None
    train: bool = False

    def __call__(self, features: list[Union[dict[str, Any], BatchEncoding]]) -> BatchEncoding:
        if self.dictionary is None:
            raise ValueError("this modules needs to dictionary.")

        features = [f.copy() for f in features]
        candidates_encodings, batch_labels = [], []
        for f in features:
            context_tokens = f.pop("input_ids")
            candidate_ids = f.pop("candidates")
            candidates = [self.dictionary(cand) for cand in candidate_ids]

            gold_ids = f.pop("labels", None)
            if self.train:
                candidates = [self.dictionary(gold_ids[0])] + candidates[:-1]
                batch_labels.append(0)
            else:
                if gold_ids:
                    gold_label_ids = [1 if cand['id'] in gold_ids else 0 for cand in candidates]
                    if 1 not in gold_label_ids:
                        batch_labels.append(-1)
                    else:
                        batch_labels.append(gold_label_ids.index(1))

            for cand in candidates:
                candidate_tokens = cand['encoding']
                concat_tokens = context_tokens + candidate_tokens + [self.tokenizer.sep_token_id]
                cand_encodings = BatchEncoding({
                    "input_ids": concat_tokens,
                    "attention_mask": [1] * len(concat_tokens),
                })
                candidates_encodings.append(cand_encodings)

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            candidates_encodings,
            max_length=self.max_length,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt"
        )

        for k, v in batch.items():
            cnum = len(v) // len(features)
            v = v.view(len(features), cnum, -1)
            batch[k] = v if self.return_tensors == "pt" else v.tolist()

        if batch_labels:
            batch["labels"] = torch.tensor(batch_labels) if self.return_tensors == "pt" else batch_labels
        return batch

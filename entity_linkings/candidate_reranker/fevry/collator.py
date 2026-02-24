import logging
from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
from transformers import BatchEncoding
from transformers.data.data_collator import pad_without_fast_tokenizer_warning

from entity_linkings.data_utils import (
    CollatorBase,
    EntityDictionary,
)

logger = logging.getLogger(__name__)


@dataclass
class CollatorForFEVRY(CollatorBase):
    dictionary: Optional[EntityDictionary] = None
    train: bool = False

    def __call__(self, features: list[Union[dict[str, Any], BatchEncoding]]) -> BatchEncoding:
        if self.dictionary is None:
            raise ValueError("this modules needs to dictionary.")

        features = [f.copy() for f in features]
        start_positions, end_positions = [], []
        batch_features, batch_candidates, batch_labels = [], [], []
        for f in features:
            new_features = {}
            new_features["input_ids"] = f.pop("input_ids")
            new_features["attention_mask"] = f.pop("attention_mask")
            token_type_ids = f.pop("token_type_ids", None)
            if token_type_ids is not None:
                new_features["token_type_ids"] = token_type_ids
            batch_features.append(new_features)

            start_position = f.pop("start_positions")
            end_position = f.pop("end_positions")
            start_positions.append(start_position)
            end_positions.append(end_position)

            golds = f.pop("labels", None)
            gold_ids = [self.dictionary.get_label_id(gold) for gold in golds] if golds is not None else []
            candidates = f.pop("candidates")
            candidate_ids = [self.dictionary.get_label_id(cand) for cand in candidates]

            if self.train:
                if not gold_ids:
                    raise ValueError("Gold labels are required during training.")
                cand_ids = [gold_ids[0]] + candidate_ids[:-1]
                batch_labels.append(0)
            else:
                cand_ids = candidate_ids
                if gold_ids:
                    gold_label_ids = [1 if cid in gold_ids else 0 for cid in cand_ids]
                    if 1 not in gold_label_ids:
                        batch_labels.append(-1)
                    else:
                        batch_labels.append(gold_label_ids.index(1))
            batch_candidates.append(cand_ids)

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            batch_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        batch["start_positions"] = torch.tensor(start_positions) if self.return_tensors == "pt" else start_positions
        batch["end_positions"] = torch.tensor(end_positions) if self.return_tensors == "pt" else end_positions
        batch["candidates_ids"] = torch.tensor(batch_candidates) if self.return_tensors == "pt" else batch_candidates
        if batch_labels:
            batch["labels"] = torch.tensor(batch_labels) if self.return_tensors == "pt" else batch_labels

        return batch

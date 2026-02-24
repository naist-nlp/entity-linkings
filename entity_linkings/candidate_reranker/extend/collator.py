from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
from transformers import BatchEncoding
from transformers.data.data_collator import pad_without_fast_tokenizer_warning

from entity_linkings.data_utils import CollatorBase, EntityDictionary


@dataclass
class CollatorForExtend(CollatorBase):
    dictionary: Optional[EntityDictionary] = None
    modify_global_attention: int = 2

    def __call__(self, features: list[Union[dict[str, Any], BatchEncoding]]) -> BatchEncoding:
        candidates_offsets = []
        starts, ends = [], []
        features = [f.copy() for f in features]
        for f in features:
            _ = f.pop("offset_mapping", None)
            _ = f.pop("labels", None)
            _ = f.pop("candidates", None)
            candidates_offsets.append(f.pop("candidates_offsets"))
            start_positions = f.pop("start_positions", None)
            end_positions = f.pop("end_positions", None)
            if start_positions is not None and end_positions is not None:
                starts.append(start_positions[0])
                ends.append(end_positions[0])

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        global_attention = torch.zeros_like(batch["attention_mask"])
        if self.modify_global_attention == 1:
            global_attention[:, 0] = 1  # CLS global attention
            for i, candidate_offsets in enumerate(candidates_offsets):
                for si, ei in candidate_offsets:
                    global_attention[i, si] = 1
                    global_attention[i, ei] = 1
        elif self.modify_global_attention == 2:
            first_candidate_starts = [
                min([si for si, _ in cand_offs])
                for cand_offs in candidates_offsets
            ]
            for i, fcs in enumerate(first_candidate_starts):
                global_attention[i, :fcs] = 1
        else:
            raise ValueError(f"Unknown modify_global_attention: {self.modify_global_attention}")

        batch["global_attention_mask"] = global_attention if self.return_tensors == "pt" else global_attention.tolist()
        if starts and ends:
            batch['start_positions'] = torch.tensor(starts) if self.return_tensors == "pt" else starts
            batch['end_positions'] = torch.tensor(ends) if self.return_tensors == "pt" else ends

        return batch

import logging
import random
from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from transformers.utils import PaddingStrategy

from .entity_dictionary import EntityDictionary

logger = logging.getLogger(__name__)


@dataclass
class CollatorBase:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        features = [f.copy() for f in features]
        encodings: list[dict[str, Any]] = []
        labels = []
        for f in features:
            encoding = f.get('encoding', None)
            if encoding is None:
                input_ids = f.get('input_ids', None)
                attention_mask = f.get('attention_mask', None)
                token_type_ids = f.get('token_type_ids', None)
                if input_ids is None or attention_mask is None:
                    raise ValueError("input_ids and attention_mask fields are required in features")
                encoding = {'input_ids': input_ids, 'attention_mask': attention_mask}
                if token_type_ids is not None:
                    encoding['token_type_ids'] = token_type_ids
            encodings.append(encoding)
            label = f.get('labels', None)
            if label is not None:
                labels.append(label)
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            encodings,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        if labels:
            batch['labels'] = labels
        return batch


@dataclass
class CollatorForRetrieval(CollatorBase):
    dictionary: Optional[EntityDictionary] = None
    num_hard_negatives: int = 0
    random_negative_sampling: bool = False

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        features = [f.copy() for f in features]
        inbatch_encodings, negative_encodings = [], []
        for f in features:
            labels = f.pop("labels")
            inbatch_encodings.append(self.dictionary(labels[0])['encoding'])
            candidates = f.pop("candidates", None)
            if candidates is not None and self.num_hard_negatives > 0:
                negative_encodings.extend([
                    self.dictionary(n)['encoding']
                    for n in candidates[:self.num_hard_negatives]
                ])

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        inbatch_candidates = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            inbatch_encodings,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch.update({f'candidates_{k}': v for k, v in inbatch_candidates.items()})
        if negative_encodings:
            negative_batch = pad_without_fast_tokenizer_warning(
                self.tokenizer,
                negative_encodings,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )
            batch.update({f'hard_negatives_{k}': v for k, v in negative_batch.items()})

        batch["labels"] = torch.arange(len(features)) if self.return_tensors == "pt" else [_ for _ in range(len(features))]

        return batch


@dataclass
class CollatorForReranking(CollatorBase):
    dictionary: Optional[EntityDictionary] = None
    train: bool = False

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
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

            golds = f.pop("labels")
            candidates = f.pop("candidates")
            gold_ids = [self.dictionary.get_label_id(gold) for gold in golds]
            candidate_ids = [self.dictionary.get_label_id(cand) for cand in candidates]
            if self.train:
                cand_ids = [gold_ids[0]] + candidate_ids
            else:
                cand_ids = candidate_ids
            batch_labels.append(gold_ids[0])
            batch_candidates.append(cand_ids)

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            batch_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        if start_positions and end_positions:
            batch["start_positions"] = torch.tensor(start_positions) if self.return_tensors == "pt" else start_positions
            batch["end_positions"] = torch.tensor(end_positions) if self.return_tensors == "pt" else end_positions

        batch["candidates_ids"] = torch.tensor(batch_candidates) if self.return_tensors == "pt" else batch_candidates
        batch["labels"] = torch.tensor(batch_labels) if self.return_tensors == "pt" else batch_labels

        return batch


@dataclass
class CollatorForCrossEncoder(CollatorBase):
    dictionary: Optional[EntityDictionary] = None
    train: bool = False

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if self.dictionary is None:
            raise ValueError("this modules needs to dictionary.")

        features = [f.copy() for f in features]
        candidates_encodings, batch_labels = [], []
        for f in features:
            context_tokens = f.pop("input_ids")
            golds = f.pop("labels")
            candidates_ids = f.pop("candidates")
            candidates = [self.dictionary(cand) for cand in candidates_ids]
            if self.train:
                candidates = [self.dictionary(golds[0])] + candidates
                batch_labels.append([1] + [0] * (len(candidates) - 1))
            else:
                gold_label_ids = [self.dictionary.get_label_id(gold) for gold in golds]
                batch_labels.append([1 if self.dictionary.get_label_id(cid) in gold_label_ids else 0 for cid in candidates_ids])

            for cand in candidates:
                candidate_tokens = cand["encoding"]
                concat_tokens = context_tokens + [self.tokenizer.sep_token_id] + candidate_tokens
                cand_encodings = self.tokenizer.prepare_for_model(
                    concat_tokens,
                    add_special_tokens=True,
                    truncation=True,
                    padding=True
                )
                if len(cand_encodings['input_ids']) == context_tokens + candidate_tokens:
                    # No special tokens were added
                    cand_encodings['input_ids'] = torch.tensor([[self.tokenizer.cls_token_id] + context_tokens + candidate_tokens + [self.tokenizer.sep_token_id]])
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

        batch["labels"] = torch.tensor(batch_labels) if self.return_tensors == "pt" else batch_labels
        return batch


@dataclass
class CollatorForExtend(CollatorBase):
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        candidates_offsets = []
        start_positions, end_positions = [], []
        features = [f.copy() for f in features]
        for f in features:
            _ = f.pop("offset_mapping", None)
            candidates_offsets.append(f.pop("candidates_offsets"))
            labels = f.pop("labels", None)
            if labels is not None:
                start_positions.append(labels[0][0])
                end_positions.append(labels[0][1])

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch["candidates_offsets"] = torch.tensor(candidates_offsets) if self.return_tensors == "pt" else candidates_offsets
        if start_positions and end_positions:
            batch['labels'] = torch.tensor([start_positions, end_positions]) if self.return_tensors == "pt" else [start_positions, end_positions]

        return batch


@dataclass
class CollatorForFusioned(CollatorBase):
    dictionary: Optional[EntityDictionary] = None
    train: bool = False

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if self.dictionary is None:
            raise ValueError("this modules needs to dictionary.")

        features = [f.copy() for f in features]
        new_features, labels = [], []
        for f in features:
            context_tokens = f.pop("input_ids")
            golds = f.pop("labels")
            labels.append(self.dictionary(golds[0])['name'])
            candidates_ids = f.pop("candidates")
            candidates = [self.dictionary(cand) for cand in candidates_ids]
            if self.train:
                candidates = [self.dictionary(golds[0])] + candidates[:-1]
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

        if labels:
            batch['labels'] = self.tokenizer(
                labels,
                padding=True,
                truncation=True,
                return_tensors=self.return_tensors,
            ).input_ids
        return batch

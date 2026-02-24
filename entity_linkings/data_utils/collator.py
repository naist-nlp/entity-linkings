import logging
from dataclasses import dataclass
from typing import Any, Optional, Union

from transformers import BatchEncoding, PreTrainedTokenizer
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from transformers.utils import PaddingStrategy

logger = logging.getLogger(__name__)


@dataclass
class CollatorBase:
    tokenizer: PreTrainedTokenizer
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: list[Union[dict[str, Any], BatchEncoding]]) -> BatchEncoding:
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

import logging
from dataclasses import dataclass
from typing import Optional

import datasets

logger = logging.getLogger(__name__)

VERSION = datasets.Version("1.0.0")

el_features = datasets.Features(
    {
        "dataset": datasets.Value("string"),
        "id": datasets.Value("string"),
        "text": datasets.Value("string"),
        "entities": [
            {
                "start": datasets.Value("int32"),
                "end": datasets.Value("int32"),
                "label": [datasets.Value("string")],
            }
        ],
    }
)

@dataclass
class EntityLinkingDatasetConfig(datasets.BuilderConfig):
    """BuilderConfig for EntityLinkingDataset."""
    name: Optional[str] = None
    version: Optional[datasets.Version] = None
    description: Optional[str] = None
    subset_id: Optional[str] = None


import logging
import os
from dataclasses import dataclass
from typing import Optional

from transformers import AutoTokenizer

from entity_linkings.data_utils import EntityDictionary

from ..dualencoder import DUALENCODER
from .encoder import TextEmbeddingModel
from .preprocessor import TextEmbeddingPreprocessor

logger = logging.getLogger(__name__)

class TEXTEMBEDDING(DUALENCODER):
    '''
    Span-level Entity Retrieval for Text Embedding Model
    '''

    @dataclass
    class Config(DUALENCODER.Config):
        '''Retrieval configuration
        Args:
            - model (str): Pre-trained model name or path for mention and entity encoders
            - start_ent_token (str): Special token to indicate the start of entity mention
            - end_ent_token (str): Special token to indicate the end of entity mention
            - entity_token (str): Special token to indicate the entity
            - nil_token (str): Special token for NIL entity
            - max_context_length (int): Maximum sequence length for mention context
            - max_candidate_length (int): Maximum sequence length for entity context
            - pooling (str): Pooling method for obtaining fixed-size representations
            - metric (str): Similarity metric for retrieval ('inner_product', 'cosine', 'euclidean')
            - loss (str): Loss function for training ('cross_entropy' or 'triplet_margin')
            - temperature (float): Temperature for scaling similarity scores
            - prefix_context (str): Prefix to add to input texts
            - prefix_candidate (str): Prefix to add to candidate texts
            - task_description (str): Task description to add to input texts
        '''
        model_name_or_path: Optional[str] = "intfloat/e5-base"
        pooling: str = 'mean'
        prefix_context: str = "query: "
        prefix_candidate: str = "passage: "
        task_description: str = ""

    def __init__(self, dictionary: EntityDictionary, config: Optional[Config] = None, index_path: Optional[str] = None) -> None:
        self.dictionary = dictionary
        self.config = config if config is not None else self.Config()

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name_or_path)
        if os.path.exists(self.config.model_name_or_path):
            logger.info(f"Loading model from {self.config.model_name_or_path}")
            self.encoder = TextEmbeddingModel.from_pretrained(self.config.model_name_or_path)
        else:
            special_tokens = [self.config.ent_start_token, self.config.ent_end_token, self.config.entity_token, self.config.nil_token]
            self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
            self.encoder = TextEmbeddingModel(
                model_name_or_path=self.config.model_name_or_path,
                pooling=self.config.pooling,
                distance=self.config.distance,
                temperature=self.config.temperature,
            )
            self.encoder.resize_token_embeddings(len(self.tokenizer))
        self.preprocessor = TextEmbeddingPreprocessor(
            self.tokenizer, self.config.ent_start_token, self.config.ent_end_token,
            self.config.entity_token, self.config.max_context_length,
            self.config.max_candidate_length, self.config.context_window_chars
        )
        self.dictionary = self.preprocessor.dictionary_preprocess(self.dictionary)
        if index_path is not None:
            self.indexer = self.create_indexer(index_path=index_path)

    def convert_to_query(self, text: str, start: int, end: int) -> str:
        marked_text = self.preprocessor._process_context(text, start, end)
        prefix = ""
        if self.config.task_description:
            prefix += f"Instruct: {self.config.task_description}\n"
        prefix += self.config.prefix_context
        return prefix + marked_text

import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Optional

import torch
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    EvalPrediction,
    set_seed,
)

from entity_linkings.data_utils import EntityDictionary
from entity_linkings.trainer import EntityLinkingTrainer, TrainingArguments
from entity_linkings.utils import BaseSystemOutput, calculate_recall_mrr

from ..base import RetrieverBase
from ..collator import CollatorForRetrieval
from .encoder import DualBERTModel
from .indexer import DenseRetriever
from .preprocessor import DualEncoderPreprocessor

logger = logging.getLogger(__name__)


def compute_metrics(p: EvalPrediction) -> dict[str, Any]:
    scores = p.predictions
    preds = scores.argmax(axis=1).ravel()
    labels = p.label_ids.ravel()
    mask = labels != -100
    preds = preds[mask]
    labels = labels[mask]

    num_corrects = (preds == labels).sum().item()
    num_golds = mask.sum().item()
    recall = num_corrects / num_golds if num_golds > 0 else float("nan")
    return {"recall": recall}


class DUALENCODER(RetrieverBase):
    '''
    Span-level Entity Retrieval with Dual-BERT Encoder
    '''

    @dataclass
    class Config(RetrieverBase.Config):
        '''Retrieval configuration
        Args:
            - model_name_or_path (str): Pre-trained model name or path for mention and entity encoders
            - ent_start_token (str): Special token to indicate the start of entity mention
            - ent_end_token (str): Special token to indicate the end of entity mention
            - entity_token (str): Special token to indicate the entity
            - nil_token (str): Special token for NIL entity
            - max_context_length (int): Maximum sequence length for mention context
            - max_candidate_length (int): Maximum sequence length for entity context
            - pooling (str): Pooling method for obtaining fixed-size representations
            - metric (str): Similarity metric for retrieval ('inner_product', 'cosine', 'euclidean')
            - num_hard_negatives (int): Number of hard negatives to sample during training
            - temperature (float): Temperature for scaling similarity scores
        '''
        model_name_or_path: Optional[str] = "google-bert/bert-base-uncased"
        ent_start_token: str = "[START_ENT]"
        ent_end_token: str = "[END_ENT]"
        entity_token: str = "[ENT]"
        nil_token: str = "[NIL]"
        max_context_length: int = 128
        max_candidate_length: int = 50
        index_batch_size: int = 128
        pooling: str = 'first'
        distance: str = 'inner_product'
        temperature: float = 1.0
        context_window_chars: int = 500
        n_hubs: int = 10
        use_hnsw: bool = False
        fp16: bool = False

    def __init__(self, dictionary: EntityDictionary, config: Optional[Config] = None, index_path: Optional[str] = None) -> None:
        super().__init__(dictionary, config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name_or_path)
        if os.path.exists(self.config.model_name_or_path):
            logger.info(f"Loading model from {self.config.model_name_or_path}")
            self.encoder = DualBERTModel.from_pretrained(self.config.model_name_or_path)
        else:
            special_tokens = [self.config.ent_start_token, self.config.ent_end_token, self.config.entity_token, self.config.nil_token]
            self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
            self.encoder = DualBERTModel(
                model_name_or_path=self.config.model_name_or_path,
                pooling=self.config.pooling,
                distance=self.config.distance,
                temperature=self.config.temperature,
            )
            self.encoder.resize_token_embeddings(len(self.tokenizer))
        self.preprocessor = DualEncoderPreprocessor(
            self.tokenizer, self.config.ent_start_token, self.config.ent_end_token,
            self.config.entity_token, self.config.max_context_length,
            self.config.max_candidate_length, self.config.context_window_chars
        )
        self.dictionary = self.preprocessor.dictionary_preprocess(self.dictionary)
        self.retriever = self.create_retriever(index_path=index_path)

    def create_retriever(self, index_path: Optional[str] = None) -> DenseRetriever:
        retriever = DenseRetriever(
            dictionary=self.dictionary,
            model=self.encoder,
            tokenizer=self.tokenizer,
            batch_size=self.config.index_batch_size,
            metric=self.config.distance,
            use_hnsw=self.config.use_hnsw,
            n_hubs=self.config.n_hubs,
            fp16=self.config.fp16
        )
        retriever.build_index(index_path=index_path)
        return retriever

    def train(
            self,
            train_dataset: Dataset,
            eval_dataset: Optional[Dataset] = None,
            num_hard_negatives: int = 0,
            training_args: Optional[TrainingArguments] = None
        ) -> dict[str, float]:
        if training_args is None:
            training_args = TrainingArguments()
        set_seed(training_args.seed)

        if num_hard_negatives > 0:
            train_candidates = self.retrieve_candidates(
                train_dataset,
                only_negative=True,
                batch_size=training_args.per_device_eval_batch_size,
                top_k=num_hard_negatives,
            )
            if eval_dataset is not None:
                eval_candidates = self.retrieve_candidates(
                    eval_dataset,
                    only_negative=True,
                    batch_size=training_args.per_device_eval_batch_size,
                    top_k=num_hard_negatives,
                )

        processed_train_dataset = self.preprocessor.dataset_preprocess(train_dataset, train_candidates if num_hard_negatives > 0 else None)
        if eval_dataset is not None:
            processed_eval_dataset = self.preprocessor.dataset_preprocess(eval_dataset, eval_candidates if num_hard_negatives > 0 else None)

        trainer = EntityLinkingTrainer(
            model=self.encoder,
            args=training_args,
            data_collator=CollatorForRetrieval(
                self.tokenizer,
                dictionary=self.dictionary,
                num_hard_negatives=num_hard_negatives
            ),
            train_dataset=processed_train_dataset,
            eval_dataset=processed_eval_dataset if eval_dataset is not None else None,
            compute_metrics=compute_metrics
        )
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint

        results = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.log_metrics("train", results.metrics)
        if training_args.output_dir is not None:
            self.encoder.save_pretrained(training_args.output_dir)
            self.tokenizer.save_pretrained(training_args.output_dir)
            trainer.save_state()
            trainer.save_metrics("train", results.metrics)
        return results

    @torch.no_grad()
    def evaluate(self, dataset: Dataset, batch_size: int = 32, **args: int) -> dict[str, float]:
        self.encoder.eval()
        queries, labels = [], []
        for text, entities in zip(dataset["text"], dataset["entities"]):
            for ent in entities:
                marked_text = self.preprocessor._process_context(text, ent["start"], ent["end"])
                queries.append(marked_text)
                labels.append(ent["label"])

        pbar = tqdm(total=(math.ceil(len(queries)//batch_size)), desc='Evaluate')
        predictions = []
        for i in range(0, len(queries), batch_size):
            pbar.update()
            _, batch_indices = self.retriever.search_knn(queries[i: i + batch_size], top_k=100)
            batch_labels = labels[i: i + batch_size]
            for j, indices in enumerate(batch_indices):
                preds = [{"id": self.dictionary(inds)["id"]} for inds in indices]
                predictions.append({"gold": batch_labels[j], "predict": preds})
        pbar.close()
        metric = calculate_recall_mrr(predictions)
        return metric

    @torch.no_grad()
    def predict(self, sentence: str, spans: Optional[list[tuple[int, int]]] = None, top_k: int = 5) -> list[list[BaseSystemOutput]]:
        if not spans:
            raise ValueError("Spans must be provided for SpanEntityRetrieval prediction.")

        self.encoder.eval()
        inputs = [self.preprocessor._process_context(sentence, b, e) for b, e in spans]
        similarities, indices = self.retriever.search_knn(inputs, top_k=top_k)

        all_result = []
        for i, (b, e) in enumerate(spans):
            result = []
            for _, ind in enumerate(indices[i]):
                entry = self.dictionary(ind)
                result.append(BaseSystemOutput(query=sentence[b:e], start=b, end=e, id=entry['id']))
            all_result.append(result)
        return all_result

    @torch.no_grad()
    def retrieve_candidates(self, dataset: Dataset, top_k: int = 5, only_negative: bool = False, batch_size: int = 32, **args: int) -> list[list[str]]:
        self.encoder.eval()
        queries, labels = [], []
        for text, entities in zip(dataset["text"], dataset["entities"]):
            for ent in entities:
                query = self.preprocessor._process_context(text, ent["start"], ent["end"])
                queries.append(query)
                labels.append(ent["label"])

        candidates = []
        pbar  = tqdm(total=len(queries), desc='Retrieve candidates')
        for i in range(0, len(queries), batch_size):
            pbar.update(min(batch_size, len(queries[i])))
            _, batch_indices = self.retriever.search_knn(
                queries[i: i + batch_size],
                top_k=top_k,
                ignore_ids=labels[i: i + batch_size] if only_negative else None
            )
            candidates.extend(batch_indices)
        pbar.close()
        return candidates

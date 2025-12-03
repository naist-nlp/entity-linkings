import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Iterator, Optional

import torch
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    BatchEncoding,
    EvalPrediction,
    set_seed,
)

from entity_linkings.data_utils import CollatorForRetrieval, cut_context_window
from entity_linkings.dataset.utils import preprocess
from entity_linkings.entity_dictionary import EntityDictionaryBase
from entity_linkings.models import EntityRetrieverBase
from entity_linkings.trainer import EntityLinkingTrainer, TrainingArguments
from entity_linkings.utils import calculate_recall_mrr

from .indexer import DenseRetriever
from .span_encoder import DualBERTModel, SpanEncoderModelBase, TextEmbeddingModel

logger = logging.getLogger(__name__)


class SpanEntityRetrievalBase(EntityRetrieverBase):
    '''
    Span-level Entity Retrieval
    '''

    @dataclass
    class Config(EntityRetrieverBase.Config):
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
        ent_start_token: str = "[START_ENT]"
        ent_end_token: str = "[END_ENT]"
        entity_token: str = "[ENT]"
        nil_token: str = "[NIL]"
        max_context_length: int = 256
        max_candidate_length: int = 50
        index_batch_size: int = 128
        pooling: str = 'first'
        distance: str = 'inner_product'
        temperature: float = 1.0
        context_window_chars: int = 500

    def __init__(self, dictionary: EntityDictionaryBase, config: Optional[Config] = None) -> None:
        super().__init__(dictionary, config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name_or_path)
        special_tokens = [self.config.ent_start_token, self.config.ent_end_token, self.config.entity_token, self.config.nil_token]
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        self.dictionary = self.dictionary_preprocess(self.dictionary)
        self.encoder = SpanEncoderModelBase(self.config.model_name_or_path)

    def compute_metrics(self, p: EvalPrediction) -> dict[str, Any]:
        loss, (_, scores) = p.predictions
        preds = scores.argmax(axis=1).ravel()
        labels = p.label_ids.ravel()
        mask = labels != -100
        preds = preds[mask]
        labels = labels[mask]

        num_corrects = (preds == labels).sum().item()
        num_golds = mask.sum().item()
        recall = num_corrects / num_golds if num_golds > 0 else float("nan")
        return {"loss": loss.sum(), "recall": recall}

    def convert_mention_template(self, text: str, start: Optional[int] = None, end: Optional[int] = None) -> str:
        return text[:start] + self.config.ent_start_token + text[start:end] + self.config.ent_end_token + text[end:]

    def convert_entity_template(self, name: str, description: str) -> str:
        return name + self.config.entity_token + description

    def create_retriever(self) -> DenseRetriever:
        retriever = DenseRetriever(
            dictionary=self.dictionary,
            config=DenseRetriever.Config(
                model=self.encoder,
                tokenizer=self.tokenizer,
                batch_size=self.config.index_batch_size,
                metric=self.config.distance,
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            )
        )
        return retriever

    def data_preprocess(self, dataset: Dataset) -> Dataset:
        def _preprocess_example(examples: Dataset) -> Iterator[BatchEncoding]:
            for text, entities in zip(examples["text"], examples["entities"]):
                for ent in entities:
                    if ent["label"]:
                        context, new_start, new_end = cut_context_window(text, ent["start"], ent["end"], self.config.context_window_chars)
                        query = self.convert_mention_template(context, new_start, new_end)
                        encodings  = self.tokenizer(query, padding=True, truncation=True, max_length=self.config.max_context_length)
                        encodings["labels"] = ent["label"]
                        yield encodings
        return preprocess(dataset, _preprocess_example)

    def dictionary_preprocess(self, dictionary: EntityDictionaryBase) -> EntityDictionaryBase:
        def preprocess_example(name: str, description: str) -> dict[str, list[int]]:
            text = self.convert_entity_template(name, description)
            encodings  = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=self.config.max_candidate_length,
            )
            return encodings
        dictionary.add_encoding(preprocess_example)
        return dictionary

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

        processed_train_dataset = self.data_preprocess(train_dataset)
        if eval_dataset is not None:
            processed_eval_dataset = self.data_preprocess(eval_dataset)
        if num_hard_negatives > 0:
            train_candidates = self.retrieve_candidates(
                train_dataset,
                only_negative=True,
                batch_size=training_args.per_device_eval_batch_size,
                top_k=num_hard_negatives,
            )
            processed_train_dataset = processed_train_dataset.add_column(name='candidates', column=train_candidates)
            if eval_dataset is not None:
                val_candidates = self.retrieve_candidates(
                    eval_dataset,
                    only_negative=True,
                    batch_size=training_args.per_device_eval_batch_size,
                    top_k=num_hard_negatives,
                )
                processed_eval_dataset = processed_eval_dataset.add_column(name='candidates', column=val_candidates)

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
            compute_metrics=self.compute_metrics
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
            retriever = self.create_retriever()
            retriever.build_index(training_args.output_dir)
        return results

    @torch.no_grad()
    def evaluate(self, dataset: Dataset, batch_size: int = 32, *args: int) -> dict[str, float]:
        self.encoder.eval()
        retriever = self.create_retriever()
        if os.path.exists(self.config.model_name_or_path):
            retriever.build_index(self.config.model_name_or_path)
        else:
            retriever.build_index()

        queries, labels = [], []
        for text, entities in zip(dataset["text"], dataset["entities"]):
            for ent in entities:
                context, new_start, new_end = cut_context_window(text, ent["start"], ent["end"], self.config.context_window_chars)
                query = self.convert_mention_template(context, new_start, new_end)
                queries.append(query)
                labels.append(ent["label"])

        pbar = tqdm(total=(math.ceil(len(queries)//batch_size)), desc='Evaluate')
        predictions = []
        for i in range(0, len(queries), batch_size):
            pbar.update()
            _, batch_indices = retriever.search_knn(queries[i: i + batch_size], top_k=100)
            batch_labels = labels[i: i + batch_size]
            for j, indices in enumerate(batch_indices):
                preds = [{"id": self.dictionary(inds)["id"]} for inds in indices]
                predictions.append({"gold": batch_labels[j], "predict": preds})
        pbar.close()
        metric = calculate_recall_mrr(predictions)
        return metric

    @torch.no_grad()
    def predict(self, sentence: str, spans: Optional[list[tuple[int, int]]] = None, top_k: int = 5) -> list[list[dict[str, Any]]]:
        if not spans:
            raise ValueError("Spans must be provided for SpanEntityRetrieval prediction.")

        self.encoder.eval()
        retriever = self.create_retriever()
        if os.path.exists(self.config.model_name_or_path):
            retriever.build_index(self.config.model_name_or_path)
        else:
            retriever.build_index()

        inputs = [self.convert_mention_template(sentence, b, e) for b, e in spans]
        similarities, indices = retriever.search_knn(inputs, top_k=top_k)

        all_result = []
        for i, (b, e) in enumerate(spans):
            result = []
            for j, ind in enumerate(indices[i]):
                entry = self.dictionary(ind)
                result.append({
                    "query": sentence[b:e],
                    "prediction": entry['name'],
                    "id": entry['id'],
                    "description": entry['description'],
                    "score": similarities[i][j]
                })
            all_result.append(result)
        return all_result

    @torch.no_grad()
    def retrieve_candidates(self, dataset: Dataset, top_k: int = 5, only_negative: bool = False, batch_size: int = 32, **args: int) -> list[list[str]]:
        self.encoder.eval()
        retriever = self.create_retriever()
        if os.path.exists(self.config.model_name_or_path):
            retriever.build_index(self.config.model_name_or_path)
        else:
            retriever.build_index()

        queries, labels = [], []
        for text, entities in zip(dataset["text"], dataset["entities"]):
            for ent in entities:
                context, new_start, new_end = cut_context_window(text, ent["start"], ent["end"], self.config.context_window_chars)
                query = self.convert_mention_template(context, new_start, new_end)
                queries.append(query)
                labels.append(ent["label"])

        candidates = []
        pbar  = tqdm(total=len(queries), desc='Retrieve candidates')
        for i in range(0, len(queries), batch_size):
            pbar.update(min(batch_size, len(queries[i])))
            _, batch_indices = retriever.search_knn(
                queries[i: i + batch_size],
                top_k=top_k,
                ignore_ids=labels[i: i + batch_size] if only_negative else None
            )
            candidates.extend(batch_indices)
        pbar.close()
        return candidates

    def load_model(self, load_directory: str) -> None:
        raise NotImplementedError


class SpanEntityRetrievalForDualEncoder(SpanEntityRetrievalBase):
    '''
    Span-level Entity Retrieval with Dual-BERT Encoder
    '''

    @dataclass
    class Config(SpanEntityRetrievalBase.Config):
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
        use_blink: bool = False

    def __init__(self, dictionary: EntityDictionaryBase, config: Optional[Config] = None) -> None:
        super().__init__(dictionary, config)
        if os.path.exists(self.config.model_name_or_path):
            print(f"Loading model from {self.config.model_name_or_path}")
            self.encoder = DualBERTModel.from_pretrained(self.config.model_name_or_path)
        else:
            self.encoder = DualBERTModel(
                model_name_or_path=self.config.model_name_or_path,
                pooling=self.config.pooling,
                distance=self.config.distance,
                temperature=self.config.temperature,
            )
            if self.config.use_blink:
                self.encoder.use_blink_weights()
            self.encoder.resize_token_embeddings(len(self.tokenizer))

    def load_model(self, load_directory: str) -> None:
        self.encoder = DualBERTModel.from_pretrained(load_directory)


class SpanEntityRetrievalForTextEmbedding(SpanEntityRetrievalBase):
    '''
    Span-level Entity Retrieval
    '''

    @dataclass
    class Config(SpanEntityRetrievalBase.Config):
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
        prefix_context: Optional[str] = "query: "
        prefix_candidate: Optional[str] = "passage: "
        task_description: Optional[str] = ""

    def __init__(self, dictionary: EntityDictionaryBase, config: Optional[Config] = None) -> None:
        super().__init__(dictionary, config)
        if os.path.exists(self.config.model_name_or_path):
            print(f"Loading model from {self.config.model_name_or_path}")
            self.encoder = TextEmbeddingModel.from_pretrained(self.config.model_name_or_path)
        else:
            self.encoder = TextEmbeddingModel(
                model_name_or_path=self.config.model_name_or_path,
                pooling=self.config.pooling,
                distance=self.config.distance,
                temperature=self.config.temperature,
            )
            self.encoder.resize_token_embeddings(len(self.tokenizer))

    def convert_mention_template(self, text: str, start: Optional[int] = None, end: Optional[int] = None) -> str:
        context = ""
        if self.config.task_description:
            context += f"Instruct: {self.config.task_description}\n"
        context += self.config.prefix_context
        context += text[:start] + self.config.ent_start_token + text[start:end] + self.config.ent_end_token + text[end:]
        return context

    def convert_entity_template(self, name: str, description: str) -> str:
        context = self.config.prefix_candidate + name + self.config.entity_token + description
        return context

    def load_model(self, load_directory: str) -> None:
        self.encoder = TextEmbeddingModel.from_pretrained(load_directory)

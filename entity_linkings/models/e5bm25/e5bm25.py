from dataclasses import dataclass
from typing import Optional

from datasets import Dataset
from transformers import set_seed

from entity_linkings.data_utils import CollatorForRetrieval, EntityDictionary
from entity_linkings.models import BM25
from entity_linkings.trainer import EntityLinkingTrainer, TrainingArguments

from ..span_retrieval import SpanEntityRetrievalForTextEmbedding


class E5BM25(SpanEntityRetrievalForTextEmbedding):
    """ E5EL: Text Embedding Models for Entity Linking
    """

    @dataclass
    class Config(SpanEntityRetrievalForTextEmbedding.Config):
        model_name_or_path: Optional[str] = "intfloat/e5-base"
        candidate_pool_size: int = 40
        random_negative_sampling: bool = True
        query_type_for_candidate: str = "mention"
        language: str = "en"

    def __init__(self, dictionary: EntityDictionary, config: Optional[Config] = None) -> None:
        super().__init__(dictionary, config)

    def train(
            self,
            train_dataset: Dataset,
            eval_dataset: Optional[Dataset] = None,
            num_hard_negatives: int = 0,
            training_args: Optional[TrainingArguments] = None
        ) -> dict[str, float]:
        if num_hard_negatives > self.config.candidate_pool_size:
            raise ValueError(
                f"The number of hard negatives ({num_hard_negatives}) "
                f"cannot be larger than the candidate pool size ({self.config.candidate_pool_size})."
            )
        if training_args is None:
            training_args = TrainingArguments()
        set_seed(training_args.seed)

        bm25_config = BM25.Config(
            model_name_or_path=training_args.output_dir,
            language=self.config.language,
            n_threads=training_args.dataloader_num_workers,
            query_type_for_candidate=self.config.query_type_for_candidate
        )
        bm25_retriever = BM25(self.dictionary, bm25_config)

        train_candidates = bm25_retriever.retrieve_candidates(
            train_dataset,
            only_negative=True,
            top_k=self.config.candidate_pool_size,
            batch_size=self.config.index_batch_size
        )
        train_dataset = self.data_preprocess(train_dataset)
        train_dataset = train_dataset.add_column("candidates", train_candidates)

        if eval_dataset is not None:
            eval_candidates = bm25_retriever.retrieve_candidates(
                eval_dataset,
                only_negative=True,
                top_k=self.config.candidate_pool_size,
                batch_size=self.config.index_batch_size
            )
            eval_dataset = self.data_preprocess(eval_dataset)
            eval_dataset = eval_dataset.add_column("candidates", eval_candidates)

        trainer = EntityLinkingTrainer(
            model=self.encoder,
            args=training_args,
            data_collator=CollatorForRetrieval(
                self.tokenizer,
                dictionary=self.dictionary,
                num_hard_negatives=num_hard_negatives,
                random_negative_sampling=self.config.random_negative_sampling,
            ),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
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

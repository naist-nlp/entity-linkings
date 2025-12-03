from importlib.resources import files

import pytest
from transformers import BertTokenizerFast
from transformers.trainer_utils import TrainOutput

import assets as test_data
from entity_linkings import load_dataset, load_dictionary
from entity_linkings.entity_dictionary.base import Entity
from entity_linkings.trainer import TrainingArguments

from .span_encoder import DualBERTModel, TextEmbeddingModel
from .span_retriever import (
    SpanEntityRetrievalBase,
    SpanEntityRetrievalForDualEncoder,
    SpanEntityRetrievalForTextEmbedding,
)

BERT_MODELS = ["google-bert/bert-base-uncased"]
TEXT_EMBEDDING_MODELS = ["intfloat/e5-base"]

dataset_path = str(files(test_data).joinpath("dataset_toy.jsonl"))
dictionary_path = str(files(test_data).joinpath("dictionary_toy.jsonl"))
dictionary = load_dictionary(dictionary_path)
dataset = load_dataset(data_files={"test": dataset_path})['test']


class TestSpanEntityRetrieverBase:
    def test__init__(self) -> None:
        model = SpanEntityRetrievalBase(
            dictionary=dictionary,
            config=SpanEntityRetrievalBase.Config(
                model_name_or_path=BERT_MODELS[0]
            )
        )
        assert isinstance(model, SpanEntityRetrievalBase)
        assert model.config.model_name_or_path == BERT_MODELS[0]
        assert model.config.ent_start_token == "[START_ENT]"
        assert model.config.ent_end_token == "[END_ENT]"
        assert model.config.entity_token == "[ENT]"
        assert model.config.nil_token == "[NIL]"
        assert model.config.max_context_length == 256
        assert model.config.max_candidate_length == 50
        assert model.config.context_window_chars == 500
        assert model.config.pooling == 'first'
        assert model.config.distance == 'inner_product'
        assert model.config.temperature == 1.0


@pytest.mark.span_retrieval_dual_encoder
class TestSpanEntityRetrieverForDualEncoder:
    def test__init__(self) -> None:
        model = SpanEntityRetrievalForDualEncoder(
            dictionary=dictionary,
            config=SpanEntityRetrievalForDualEncoder.Config(model_name_or_path=BERT_MODELS[0])
        )
        assert isinstance(model, SpanEntityRetrievalForDualEncoder)
        assert model.config.ent_start_token in model.tokenizer.all_special_tokens
        assert model.config.ent_start_token in model.tokenizer.all_special_tokens
        assert model.config.entity_token in model.tokenizer.all_special_tokens
        assert model.config.nil_token in model.tokenizer.all_special_tokens
        assert isinstance(model.encoder, DualBERTModel)
        assert isinstance(model.tokenizer, BertTokenizerFast)
        assert model.config.model_name_or_path == BERT_MODELS[0]
        for processed in model.dictionary:
            assert "encoding" in processed
            assert "input_ids" in processed["encoding"]
            assert "attention_mask" in processed["encoding"]

    def test_convert_mention_template(self) -> None:
        model = SpanEntityRetrievalForDualEncoder(
            dictionary=dictionary,
            config=SpanEntityRetrievalForDualEncoder.Config(model_name_or_path=BERT_MODELS[0])
        )
        example = "This is a mention in the text."
        converted = model.convert_mention_template(example, start=10, end=17)
        assert converted == "This is a [START_ENT]mention[END_ENT] in the text."

    def test_convert_entity_template(self) -> None:
        model = SpanEntityRetrievalForDualEncoder(
            dictionary=dictionary,
            config=SpanEntityRetrievalForDualEncoder.Config(model_name_or_path=BERT_MODELS[0])
        )
        example = Entity(id="1", name="Test", description="This is a test entity.", label_id=0, encoding=None)
        converted = model.convert_entity_template(example['name'], example['description'])
        assert converted == "Test[ENT]This is a test entity."

    def test_data_preprocess(self) -> None:
        model = SpanEntityRetrievalForDualEncoder(
            dictionary=dictionary,
            config=SpanEntityRetrievalForDualEncoder.Config(
                model_name_or_path=BERT_MODELS[0]
            )
        )
        processed_dataset = model.data_preprocess(dataset)
        assert len(processed_dataset) == 8
        for processed in processed_dataset:
            assert "input_ids" in processed
            assert "attention_mask" in processed

    def test_train(self) -> None:
        model = SpanEntityRetrievalForDualEncoder(dictionary=dictionary)
        result = model.train(
            train_dataset=dataset,
            eval_dataset=dataset,
            num_hard_negatives=2,
            training_args=TrainingArguments(
                output_dir="./test_output",
                num_train_epochs=1,
                per_device_train_batch_size=2,
                per_device_eval_batch_size=2,
                logging_strategy="no",
                save_strategy="no",
                eval_strategy="no",
                remove_unused_columns=False,
                eval_on_start=True
            )
        )
        assert isinstance(result, TrainOutput)
        assert hasattr(result, 'metrics')

    def test_evaluate(self) -> None:
        model = SpanEntityRetrievalForDualEncoder(dictionary=dictionary)
        metrics = model.evaluate(dataset)
        assert 'recall@1' in metrics
        assert 'recall@10' in metrics
        assert 'recall@50' in metrics
        assert 'recall@100' in metrics
        assert 'mrr@100' in metrics

    def test_predict(self) -> None:
        model = SpanEntityRetrievalForDualEncoder(dictionary=dictionary)
        sentence = "Steve Jobs was found Apple."
        spans = [(21, 26)]
        top_k = 3
        predictions = model.predict(sentence, spans, top_k)
        assert isinstance(predictions, list)
        assert len(predictions) == len(spans)
        for preds in predictions:
            assert isinstance(preds, list)
            assert len(preds) == min(top_k, len(dictionary))

    def test_retrieve_candidates(self) -> None:
        model = SpanEntityRetrievalForDualEncoder(dictionary=dictionary)
        top_k = 3
        candidate_lists = model.retrieve_candidates(dataset, top_k=top_k, batch_size=1, negative=True)
        assert isinstance(candidate_lists, list)
        assert len(candidate_lists) == len(dataset)
        for candidates in candidate_lists:
            assert isinstance(candidates, list)
            assert len(candidates) == top_k
            for candidate in candidates:
                assert isinstance(candidate, str)

@pytest.mark.span_retrieval_text_embedding
class TestSpanEntityRetrieverForTextEmbedding:
    def test__init__(self) -> None:
        model = SpanEntityRetrievalForTextEmbedding(
            dictionary=dictionary,
            config=SpanEntityRetrievalForTextEmbedding.Config(
                model_name_or_path=TEXT_EMBEDDING_MODELS[0]
            )
        )
        assert isinstance(model, SpanEntityRetrievalForTextEmbedding)
        assert isinstance(model.tokenizer, BertTokenizerFast)
        assert isinstance(model.encoder, TextEmbeddingModel)
        assert model.config.ent_start_token in model.tokenizer.all_special_tokens
        assert model.config.ent_start_token in model.tokenizer.all_special_tokens
        assert model.config.entity_token in model.tokenizer.all_special_tokens
        assert model.config.nil_token in model.tokenizer.all_special_tokens
        assert model.config.model_name_or_path == TEXT_EMBEDDING_MODELS[0]
        for processed in model.dictionary:
            assert "encoding" in processed
            assert "input_ids" in processed["encoding"]
            assert "attention_mask" in processed["encoding"]

    def test_convert_mention_template(self) -> None:
        model = SpanEntityRetrievalForTextEmbedding(
            dictionary=dictionary,
            config=SpanEntityRetrievalForTextEmbedding.Config(
                model_name_or_path=TEXT_EMBEDDING_MODELS[0],
                task_description="Entity linking task",
            )
        )
        example = "This is a mention in the text."
        converted = model.convert_mention_template(example, start=10, end=17)
        assert converted == "Instruct: Entity linking task\nquery: This is a [START_ENT]mention[END_ENT] in the text."

    def test_convert_entity_template(self) -> None:
        model = SpanEntityRetrievalForTextEmbedding(
            dictionary=dictionary,
            config=SpanEntityRetrievalForTextEmbedding.Config(
                model_name_or_path=TEXT_EMBEDDING_MODELS[0],
            )
        )
        example = Entity(id="1", name="Test", description="This is a test entity.", label_id=0, encoding=None)
        converted = model.convert_entity_template(example['name'], example['description'])
        assert converted == "passage: Test[ENT]This is a test entity."

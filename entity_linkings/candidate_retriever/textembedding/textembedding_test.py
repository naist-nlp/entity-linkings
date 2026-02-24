import tempfile
from importlib.resources import files

import pytest
from datasets import load_dataset
from transformers import BertTokenizerFast
from transformers.trainer_utils import TrainOutput

import assets as test_data
from entity_linkings import load_dictionary
from entity_linkings.trainer import TrainingArguments

from .encoder import TextEmbeddingModel
from .textembedding import TEXTEMBEDDING

TEXT_EMBEDDING_MODELS = ["intfloat/e5-base"]

dataset_path = str(files(test_data).joinpath("dataset_toy.jsonl"))
dictionary_path = str(files(test_data).joinpath("dictionary_toy.jsonl"))
dictionary = load_dictionary(dictionary_path)
dataset = load_dataset("json", data_files={"test": dataset_path})['test']

@pytest.mark.span_retrieval_text_embedding
class TestSpanEntityRetrieverForTextEmbedding:
    def test__init__(self) -> None:
        model = TEXTEMBEDDING(
            dictionary=dictionary,
            config=TEXTEMBEDDING.Config(
                model_name_or_path=TEXT_EMBEDDING_MODELS[0]
            )
        )
        assert isinstance(model, TEXTEMBEDDING)
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

    def test_train(self) -> None:
        model = TEXTEMBEDDING(dictionary=dictionary)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = model.train(
                train_dataset=dataset,
                eval_dataset=dataset,
                num_hard_negatives=2,
                training_args=TrainingArguments(
                    output_dir=tmpdir,
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
        model = TEXTEMBEDDING(dictionary=dictionary)
        metrics = model.evaluate(dataset)
        assert 'recall@1' in metrics
        assert 'recall@10' in metrics
        assert 'recall@50' in metrics
        assert 'recall@100' in metrics
        assert 'mrr@100' in metrics

    def test_predict(self) -> None:
        model = TEXTEMBEDDING(dictionary=dictionary)
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
        model = TEXTEMBEDDING(dictionary=dictionary)
        top_k = 3
        candidate_lists = model.retrieve_candidates(dataset, top_k=top_k, batch_size=1, negative=True)
        assert isinstance(candidate_lists, list)
        assert len(candidate_lists) == len(dataset)
        for candidates in candidate_lists:
            assert isinstance(candidates, list)
            assert len(candidates) == top_k
            for candidate in candidates:
                assert isinstance(candidate, str)

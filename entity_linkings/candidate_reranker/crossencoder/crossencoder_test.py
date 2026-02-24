import tempfile
from importlib.resources import files

import pytest
from datasets import load_dataset
from transformers.trainer_utils import TrainOutput

import assets as test_data
from entity_linkings import get_rerankers, get_retrievers, load_dictionary
from entity_linkings.trainer import TrainingArguments
from entity_linkings.utils import BaseSystemOutput

from .crossencoder import CROSSENCODER

MODELS = ["google-bert/bert-base-uncased"]
dataset_path = str(files(test_data).joinpath("dataset_toy_wo_candidates.jsonl"))
dictionary_path = str(files(test_data).joinpath("dictionary_toy.jsonl"))
dictionary = load_dictionary(dictionary_path)
dataset = load_dataset("json", data_files={"test": dataset_path})['test']
retriever_cls = get_retrievers("bm25")
retriever = retriever_cls(dictionary=dictionary)


@pytest.mark.reranker_crossencoder
class TestCROSSENCODER:
    def test__init__(self) -> None:
        model_cls = get_rerankers("crossencoder")
        model = model_cls(
            retriever=retriever,
            config=model_cls.Config(
                model_name_or_path=MODELS[0]
            )
        )
        assert isinstance(model, CROSSENCODER)
        assert hasattr(model, "config") and hasattr(model, "tokenizer") and hasattr(model, "dictionary")
        assert model.config.model_name_or_path == "google-bert/bert-base-uncased"
        assert model.config.nil_token == "[NIL]" and model.config.nil_token in model.tokenizer.all_special_tokens
        assert model.config.ent_start_token == "[START_ENT]" and model.config.ent_start_token in model.tokenizer.all_special_tokens
        assert model.config.ent_end_token == "[END_ENT]" and model.config.ent_end_token in model.tokenizer.all_special_tokens
        assert model.config.entity_token == "[ENT]" and model.config.entity_token in model.tokenizer.all_special_tokens
        assert model.config.max_context_length == 128
        assert model.config.max_candidate_length == 50
        assert model.config.context_window_chars == 500
        assert model.config.pooling == 'first'

    def test_train(self) -> None:
        model = CROSSENCODER(retriever=retriever, config=CROSSENCODER.Config(model_name_or_path=MODELS[0]))
        with tempfile.TemporaryDirectory() as tmpdir:
            result = model.train(
                    train_dataset=dataset,
                    eval_dataset=dataset,
                    num_candidates=3,
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
        model = CROSSENCODER(retriever=retriever, config=CROSSENCODER.Config(model_name_or_path=MODELS[0]))
        metrics = model.evaluate(dataset, num_candidates=3, batch_size=1)
        assert 'top1_accuracy' in metrics

    def test_predict(self) -> None:
        sentence = "Apple is looking at buying U.K. startup for $1 billion"
        spans = [(0,5), (27,30), (44,52)]
        model = CROSSENCODER(retriever=retriever, config=CROSSENCODER.Config(model_name_or_path=MODELS[0]))
        predictions = model.predict(sentence, spans=spans, num_candidates=3)
        assert isinstance(predictions, list)
        assert len(predictions) == len(spans)
        for preds in predictions:
            assert isinstance(preds, BaseSystemOutput)
            assert preds.id is not None and preds.start is not None and preds.end is not None


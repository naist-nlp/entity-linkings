from importlib.resources import files

import pytest
from transformers.trainer_utils import TrainOutput

import assets as test_data
from entity_linkings import get_models, get_retrievers, load_dataset, load_dictionary
from entity_linkings.entity_dictionary.base import Entity
from entity_linkings.trainer import TrainingArguments

from .blink import BLINK

MODELS = ["google-bert/bert-base-uncased"]
dataset_path = str(files(test_data).joinpath("dataset_toy_wo_candidates.jsonl"))
dictionary_path = str(files(test_data).joinpath("dictionary_toy.jsonl"))
dictionary = load_dictionary(dictionary_path)
dataset = load_dataset(data_files={"test": dataset_path})['test']
retriever_cls = get_retrievers("bm25")
retriever = retriever_cls(dictionary=dictionary)
candidates_ids = retriever.retrieve_candidates(dataset, top_k=3)

@pytest.mark.disambiguate_blink
class TestBLINK:
    def test__init__(self) -> None:
        model_cls = get_models("blink")
        model = model_cls(
            retriever=retriever,
            config=model_cls.Config(
                model_name_or_path=MODELS[0]
            )
        )
        assert isinstance(model, BLINK)
        assert hasattr(model, "config") and hasattr(model, "tokenizer") and hasattr(model, "dictionary")
        assert model.config.num_candidates == 30
        assert model.config.model_name_or_path == "google-bert/bert-base-uncased"
        assert model.config.ent_start_token == "[START_ENT]"
        assert model.config.ent_end_token == "[END_ENT]"
        assert model.config.entity_token == "[ENT]"
        assert model.config.nil_token == "[NIL]"
        assert model.config.max_context_length == 128
        assert model.config.max_candidate_length == 50
        assert model.config.pooling == 'first'

    def test_convert_mention_template(self) -> None:
        model = BLINK(retriever=retriever, config=BLINK.Config(model_name_or_path=MODELS[0]))
        example = "This is a mention in the text."
        converted = model.convert_mention_template(example, start=10, end=17)
        assert converted == "This is a [START_ENT]mention[END_ENT] in the text."

    def test_convert_entity_template(self) -> None:
        model = BLINK(retriever=retriever, config=BLINK.Config(model_name_or_path=MODELS[0]))
        example = Entity(id="1", name="Test", description="This is a test entity.", label_id=0, encoding=None)
        converted = model.convert_entity_template(example['name'], example['description'])
        assert converted == "Test[ENT]This is a test entity."

    def test_data_preprocess(self) -> None:
        model = BLINK(retriever=retriever, config=BLINK.Config(model_name_or_path=MODELS[0]))
        processed_dataset = model.data_preprocess(dataset)
        assert len(processed_dataset) == 8
        for processed in processed_dataset:
            assert "input_ids" in processed
            assert "attention_mask" in processed
            assert "labels" in processed

    def test_train(self) -> None:
        model = BLINK(retriever=retriever, config=BLINK.Config(model_name_or_path=MODELS[0]))
        result = model.train(
            train_dataset=dataset,
            eval_dataset=dataset,
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
        model = BLINK(retriever=retriever, config=BLINK.Config(model_name_or_path=MODELS[0]))
        metrics = model.evaluate(dataset, num_candidates=3, batch_size=1)
        assert 'top1_accuracy' in metrics

    def test_predict(self) -> None:
        sentence = "Apple is looking at buying U.K. startup for $1 billion"
        spans = [(0,5), (27,30), (44,52)]
        model = BLINK(retriever=retriever, config=BLINK.Config(model_name_or_path=MODELS[0]))
        predictions = model.predict(sentence, spans=spans, num_candidates=2)
        assert isinstance(predictions, list)
        assert len(predictions) == len(spans)
        for preds in predictions:
            assert isinstance(preds, list)
            assert len(preds) == 2

import tempfile
from importlib.resources import files

import pytest
from transformers.trainer_utils import TrainOutput

import assets as test_data
from entity_linkings import get_models, get_retrievers, load_dataset, load_dictionary
from entity_linkings.trainer import TrainingArguments

from .fusioned import FUSIONED

MODELS = ["google/flan-t5-base"]
dataset_path = str(files(test_data).joinpath("dataset_toy_wo_candidates.jsonl"))
dictionary_path = str(files(test_data).joinpath("dictionary_toy.jsonl"))
dictionary = load_dictionary(dictionary_path)
dataset = load_dataset(data_files={"test": dataset_path})['test']
retriever_cls = get_retrievers("bm25")
retriever = retriever_cls(dictionary=dictionary)
candidates_ids = retriever.retrieve_candidates(dataset, top_k=3)


@pytest.mark.disambiguate_fusioned
class TestFUSIONED:
    def test__init__(self) -> None:
        model_cls = get_models("fusioned")
        model = model_cls(retriever=retriever)
        assert isinstance(model, FUSIONED)
        assert hasattr(model, "config") and hasattr(model, "tokenizer") and hasattr(model, "dictionary")
        assert model.config.model_name_or_path == "google/flan-t5-base"
        assert model.config.num_candidates == 30
        assert model.config.max_context_length == 250
        assert model.config.max_candidate_length == 140
        assert model.config.document_token == "<extra_id_0>"
        assert model.config.passage_token == "<extra_id_1>"
        assert model.config.title_token == "<extra_id_2>"
        assert model.config.description_token == "<extra_id_3>"
        assert model.config.entity_token == "<extra_id_4>"
        assert model.config.mention_token == "<extra_id_5>"
        assert model.config.ent_start_token == "<extra_id_6>"
        assert model.config.ent_end_token == "<extra_id_7>"
        for d in model.dictionary:
            assert "encoding" in d

    def test_prosess_text(self) -> None:
        model = FUSIONED(
            retriever=retriever,
            config=FUSIONED.Config(model_name_or_path=MODELS[0])
        )
        encodings = model.process_context(dataset['text'][0], dataset['entities'][0][0]['start'], dataset['entities'][0][0]['end'])
        assert "input_ids" in encodings
        assert "attention_mask" in encodings
        assert len(encodings["input_ids"]) == len(encodings["attention_mask"])
        assert len(encodings["input_ids"]) <= model.config.max_context_length

    def test_data_preprocess(self) -> None:
        model = FUSIONED(
            retriever=retriever,
            config=FUSIONED.Config(model_name_or_path=MODELS[0])
        )
        processed_dataset = model.data_preprocess(dataset)
        assert len(processed_dataset) == 8
        for processed in processed_dataset:
            assert "input_ids" in processed
            assert "attention_mask" in processed
            assert "labels" in processed

    def test_train(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = FUSIONED(
                retriever=retriever,
                config=FUSIONED.Config(model_name_or_path=MODELS[0])
            )
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
        model = FUSIONED(
            retriever=retriever,
            config=FUSIONED.Config(model_name_or_path=MODELS[0])
        )
        metrics = model.evaluate(dataset, num_candidates=3, batch_size=1)
        assert 'top1_accuracy' in metrics

    def test_predict(self) -> None:
        sentence = "Apple is looking at buying U.K. startup for $1 billion"
        spans = [(0,5), (27,30), (44,52)]
        model = FUSIONED(
            retriever=retriever,
            config=FUSIONED.Config(model_name_or_path=MODELS[0])
        )
        predictions = model.predict(sentence, spans=spans, num_candidates=2)
        assert isinstance(predictions, list)
        assert len(predictions) == len(spans)
        for preds in predictions:
            assert isinstance(preds, list)

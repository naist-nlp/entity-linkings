import tempfile
from importlib.resources import files

import pytest
from datasets import load_dataset
from transformers.trainer_utils import TrainOutput

import assets as test_data
from entity_linkings import get_rerankers, get_retrievers, load_dictionary
from entity_linkings.trainer import TrainingArguments
from entity_linkings.utils import BaseSystemOutput

from .fusioned import FUSIONED

MODELS = ["google/flan-t5-base"]
dataset_path = str(files(test_data).joinpath("dataset_toy_wo_candidates.jsonl"))
dictionary_path = str(files(test_data).joinpath("dictionary_toy.jsonl"))
dictionary = load_dictionary(dictionary_path)
dataset = load_dataset("json", data_files={"test": dataset_path})['test']
retriever_cls = get_retrievers("bm25")
retriever = retriever_cls(dictionary=dictionary)


@pytest.mark.reranker_fusioned
class TestFUSIONED:
    @pytest.mark.parametrize("use_checkpoint", [True, False])
    def test__init__(self, use_checkpoint: bool) -> None:
        model_cls = get_rerankers("fusioned")
        model = model_cls(retriever=retriever)
        assert isinstance(model, FUSIONED)
        assert hasattr(model, "config") and hasattr(model, "tokenizer") and hasattr(model, "dictionary")
        assert model.config.model_name_or_path == "google/flan-t5-base"
        assert model.config.max_context_length == 128
        assert model.config.max_candidate_length == 50
        assert model.config.context_window_chars == 500
        assert model.config.use_checkpoint == False if not use_checkpoint else True
        assert model.config.num_beams == 3
        assert model.config.max_new_tokens == 200
        assert model.config.min_length == 1
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

    @pytest.mark.parametrize("use_checkpoint", [True, False])
    def test_train(self, use_checkpoint: bool) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = FUSIONED(
                retriever=retriever,
                config=FUSIONED.Config(
                    model_name_or_path=MODELS[0],
                    use_checkpoint=use_checkpoint
                )
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

    @pytest.mark.parametrize("use_checkpoint", [True, False])
    def test_evaluate(self, use_checkpoint: bool) -> None:
        model = FUSIONED(
            retriever=retriever,
            config=FUSIONED.Config(model_name_or_path=MODELS[0], use_checkpoint=use_checkpoint)
        )
        metrics = model.evaluate(dataset, num_candidates=3, batch_size=1)
        assert 'top1_accuracy' in metrics

    @pytest.mark.parametrize("use_checkpoint", [True, False])
    def test_predict(self, use_checkpoint: bool) -> None:
        sentence = "Apple is looking at buying U.K. startup for $1 billion"
        spans = [(0,5), (27,30), (44,52)]
        model = FUSIONED(
            retriever=retriever,
            config=FUSIONED.Config(model_name_or_path=MODELS[0], use_checkpoint=use_checkpoint)
        )
        predictions = model.predict(sentence, spans=spans, num_candidates=3)
        assert isinstance(predictions, list)
        for preds in predictions:
            assert isinstance(preds, BaseSystemOutput)
            assert preds.id is not None and preds.start is not None and preds.end is not None


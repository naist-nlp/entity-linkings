from importlib.resources import files

import pytest
from transformers.trainer_utils import TrainOutput

import assets as test_data
from entity_linkings import load_dataset, load_dictionary
from entity_linkings.models import E5BM25
from entity_linkings.models.span_retrieval.span_encoder import TextEmbeddingModel
from entity_linkings.trainer import TrainingArguments

MODELS = ["intfloat/e5-base"]

dataset_path = str(files(test_data).joinpath("dataset_toy_wo_candidates.jsonl"))
dictionary_path = str(files(test_data).joinpath("dictionary_toy.jsonl"))
dictionary = load_dictionary(dictionary_path)
dataset = load_dataset(data_files={"test": dataset_path})['test']


@pytest.mark.span_retrieval_e5bm25
class TestE5BM25:
    @pytest.mark.parametrize("model_name", MODELS)
    def test__init__(self, model_name: str) -> None:
        model = E5BM25(
            dictionary=dictionary,
            config=E5BM25.Config(model_name_or_path=model_name)
        )
        assert isinstance(model, E5BM25)
        assert isinstance(model.encoder, TextEmbeddingModel)
        assert model.config.candidate_pool_size == 40
        assert model.config.random_negative_sampling is True
        assert model.config.model_name_or_path == model_name

    @pytest.mark.parametrize("model_name", MODELS)
    def test_train(self, model_name: str) -> None:
        model = E5BM25(
            dictionary=dictionary,
            config=E5BM25.Config(
                candidate_pool_size=3,
                model_name_or_path=model_name
            )
        )
        result = model.train(
            train_dataset=dataset,
            eval_dataset=dataset,
            num_hard_negatives=1,
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

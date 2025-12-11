import tempfile
from importlib.resources import files

import pytest
from transformers.trainer_utils import TrainOutput

import assets as test_data
from entity_linkings import get_models, load_dataset, load_dictionary
from entity_linkings.models import BM25
from entity_linkings.trainer import TrainingArguments

from .extend import EXTEND

MODELS = ["google-bert/bert-base-uncased"]
dictionary_path = str(files(test_data).joinpath("dictionary_toy.jsonl"))
dictionary = load_dictionary(dictionary_path)
dataset_path = str(files(test_data).joinpath("dataset_toy.jsonl"))
dataset = load_dataset("json", dataset_path, split="train", cache_dir='.cache_test')
retriever = BM25(dictionary)


@pytest.mark.disambiguate_extend
class TestEXTEND:
    def test___init__(self) -> None:
        model_cls = get_models("extend")
        model = model_cls(
            retriever=retriever,
            config=EXTEND.Config(model_name_or_path=MODELS[0])
        )
        assert isinstance(model, EXTEND)
        assert hasattr(model, "config") and hasattr(model, "tokenizer") and hasattr(model, "dictionary")
        assert model.config.model_name_or_path == MODELS[0]
        assert model.config.ent_start_token in model.tokenizer.all_special_tokens
        assert model.config.ent_end_token in model.tokenizer.all_special_tokens
        assert model.config.nil_token in model.tokenizer.all_special_tokens
        assert model.config.max_context_length == 128
        assert model.config.attention_window == 64
        assert model.config.modify_global_attention == 2
        assert model.config.mode == "max-prod"

    def test_data_filter(self) -> None:
        model = EXTEND(retriever=retriever, config=EXTEND.Config(model_name_or_path=MODELS[0]))
        candidates = retriever.retrieve_candidates(dataset, top_k=3, only_negative=True)
        filtered_dataset = model.data_filter(dataset)
        assert len(filtered_dataset) == len(candidates) == 8
        for filtered in filtered_dataset:
            assert "text" in filtered
            assert "start" in filtered
            assert "end" in filtered
            assert "label" in filtered

    def test_process_context(self) -> None:
        model = EXTEND(retriever=retriever, config=EXTEND.Config(model_name_or_path=MODELS[0]))
        candidates = retriever.retrieve_candidates(dataset, top_k=3, only_negative=True)
        filtered_dataset = model.data_filter(dataset)
        filtered_dataset = filtered_dataset.add_column("candidates", candidates)
        for filtered in filtered_dataset:
            encodings = model.process_context(filtered['text'], filtered['start'], filtered['end'], filtered['candidates'], filtered['label'])
            assert "input_ids" in encodings
            assert "attention_mask" in encodings
            assert "offset_mapping" in encodings
            assert "labels" in encodings

    @pytest.mark.parametrize("train", [True, False])
    def test_data_preprocess(self, train: bool) -> None:
        model = EXTEND(retriever=retriever, config=EXTEND.Config(model_name_or_path=MODELS[0]))
        candidates = retriever.retrieve_candidates(dataset, top_k=3, only_negative=True)
        filtered_dataset = model.data_filter(dataset)
        filtered_dataset = filtered_dataset.add_column("candidates", candidates)
        processed_dataset = model.data_preprocess(filtered_dataset, train=train)
        assert len(processed_dataset) ==  8
        for processed in processed_dataset:
            assert "input_ids" in processed
            assert "attention_mask" in processed
            assert "candidates_offsets" in processed
            if train:
                assert "labels" in processed
                assert processed["labels"][0] in processed["candidates_offsets"]
                assert len(processed["labels"]) == 1
                assert len(processed["labels"][0]) == 2
            else:
                assert "labels" not in processed

    def test_train(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = EXTEND(retriever=retriever, config=EXTEND.Config(model_name_or_path=MODELS[0]))
            result = model.train(
                train_dataset=dataset,
                eval_dataset=dataset,
                num_candidates=3,
                training_args=TrainingArguments(
                    output_dir=tmpdir,
                    num_train_epochs=1,
                    per_device_train_batch_size=3,
                    per_device_eval_batch_size=4,
                    logging_strategy="no",
                    save_strategy="no",
                    eval_strategy="no",
                    remove_unused_columns=False,
                    eval_on_start=False
                )
            )
            assert isinstance(result, TrainOutput)
            assert hasattr(result, 'metrics')

    def test_evaluate(self) -> None:
        model = EXTEND(retriever=retriever, config=EXTEND.Config(model_name_or_path=MODELS[0]))
        metrics = model.evaluate(dataset, num_candidates=3, batch_size=4)
        assert 'top1_accuracy' in metrics

    def test_predict(self) -> None:
        sentence = "Apple is looking at buying U.K. startup for $1 billion"
        spans = [(0,5), (27,30), (44,52)]
        model = EXTEND(retriever=retriever, config=EXTEND.Config(model_name_or_path=MODELS[0]))
        predictions = model.predict(sentence, spans=spans, num_candidates=2)
        assert isinstance(predictions, list)
        assert len(predictions) == len(spans)
        for preds in predictions:
            assert isinstance(preds, list)

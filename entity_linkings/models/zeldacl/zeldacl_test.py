import tempfile
from importlib.resources import files

import assets as test_data
from entity_linkings import load_dataset, load_dictionary

from .zeldacl import ZELDACL

dataset_path = str(files(test_data).joinpath("dataset_toy.jsonl"))
dictionary_path = str(files(test_data).joinpath("dictionary_toy.jsonl"))
dictionary = load_dictionary(dictionary_path)
dataset = load_dataset(data_files={"test": dataset_path})['test']


class TestZELDACL:
    def test_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ZELDACL.Config(model_name_or_path=tmpdir)
            assert config.model_name_or_path == tmpdir

    def test_init(self) -> None:
        zeldacl_model = ZELDACL(
            dictionary=dictionary,
            config=ZELDACL.Config()
        )
        assert isinstance(zeldacl_model, ZELDACL)
        assert zeldacl_model.retriever is not None
        assert zeldacl_model.dictionary is not None

    def test_evaluate(self) -> None:
        zeldacl_model = ZELDACL(
            dictionary=dictionary,
            config=ZELDACL.Config()
        )
        metrics = zeldacl_model.evaluate(dataset)
        assert 'recall@1' in metrics
        assert 'recall@10' in metrics
        assert 'recall@50' in metrics
        assert 'recall@100' in metrics
        assert 'mrr@100' in metrics

    def test_predict(self) -> None:
        zeldacl_model = ZELDACL(
            dictionary=dictionary,
            config=ZELDACL.Config()
        )
        sentence = "Steve Jobs was found Apple."
        spans = [(21, 26)]
        top_k = 3
        predictions = zeldacl_model.predict(sentence, spans, top_k)
        assert isinstance(predictions, list)
        assert len(predictions) == len(spans)
        for preds in predictions:
            assert isinstance(preds, list)
            assert len(preds) == min(top_k, len(dictionary))

    def test_retrieve_candidates(self) -> None:
        zeldacl_model = ZELDACL(
            dictionary=dictionary,
            config=ZELDACL.Config()
        )
        top_k = 5
        candidates = zeldacl_model.retrieve_candidates(dataset, top_k=top_k, negative=False, batch_size=3)
        assert isinstance(candidates, list)
        total_entities = sum(len(example["entities"]) for example in dataset)
        assert len(candidates) == total_entities
        for inds in candidates:
            assert isinstance(inds, list)
            assert len(inds) == top_k

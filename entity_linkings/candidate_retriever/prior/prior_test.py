from importlib.resources import files

import pytest
from datasets import load_dataset

import assets as test_data
from entity_linkings import load_dictionary
from entity_linkings.utils import BaseSystemOutput

from .prior import PRIOR

dataset_path = str(files(test_data).joinpath("dataset_toy.jsonl"))
dictionary_path = str(files(test_data).joinpath("dictionary_toy.jsonl"))
dictionary = load_dictionary(dictionary_path)
dataset = load_dataset("json", data_files={"test": dataset_path})['test']
mention_counter_path = str(files(test_data).joinpath("mention_counter_toy.json"))


@pytest.fixture(scope='module')
def prior_model() -> PRIOR:
    model = PRIOR(
        dictionary=dictionary,
        config=PRIOR.Config(model_name_or_path=mention_counter_path)
    )
    return model


@pytest.mark.retriever_prior
class TestPrior:
    def test_init(self, prior_model: PRIOR) -> None:
        assert isinstance(prior_model, PRIOR)
        assert prior_model.retriever is not None
        assert prior_model.dictionary is not None

    def test_evaluate(self, prior_model: PRIOR) -> None:
        metrics = prior_model.evaluate(dataset)
        assert 'recall@1' in metrics
        assert 'recall@10' in metrics
        assert 'recall@50' in metrics
        assert 'recall@100' in metrics
        assert 'mrr@100' in metrics

    def test_predict(self, prior_model: PRIOR) -> None:
        sentence = "Steve Jobs was found Apple."
        spans = [(21, 26)]
        top_k = 3
        predictions = prior_model.predict(sentence, spans, top_k)
        assert isinstance(predictions, list)
        assert len(predictions) == len(spans)
        for preds in predictions:
            assert isinstance(preds, list)
            assert len(preds) == min(top_k, len(dictionary))
            for pred in preds:
                assert isinstance(pred, BaseSystemOutput)
                assert pred.id is not None and pred.start is not None and pred.end is not None


    def test_retrieve_candidates(self, prior_model: PRIOR) -> None:
        top_k = 5
        candidates = prior_model.retrieve_candidates(dataset, top_k=top_k, negative=False, batch_size=3)
        assert isinstance(candidates, list)
        total_entities = sum(len(example["entities"]) for example in dataset)
        assert len(candidates) == total_entities
        for inds in candidates:
            assert isinstance(inds, list)
            assert len(inds) == top_k

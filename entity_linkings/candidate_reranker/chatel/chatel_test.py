from importlib.resources import files

import pytest
from datasets import load_dataset

import assets as test_data
from entity_linkings import get_retrievers, load_dictionary
from entity_linkings.utils import BaseSystemOutput

from .chatel import CHATEL
from .utils import process_multi_choice_prompt

MODELS = ["gpt-3.5-turbo"]
dictionary_path = str(files(test_data).joinpath("dictionary_toy.jsonl"))
dictionary = load_dictionary(dictionary_path)
dataset_path = str(files(test_data).joinpath("dataset_toy.jsonl"))
dataset = load_dataset("json", data_files={"test": dataset_path})['test']
retriever_cls = get_retrievers("bm25")
retriever = retriever_cls(dictionary=dictionary)
candidates_ids = retriever.retrieve_candidates(dataset, top_k=3)


@pytest.mark.reranker_chatel
class TestChatEL:
    @pytest.fixture
    def model(self) -> CHATEL:
        return CHATEL(
            retriever=retriever,
            config=CHATEL.Config(
                model_name_or_path=MODELS[0],
                token="dummy-key"
            ),
        )

    def test__init__(self, model: CHATEL) -> None:
        assert hasattr(model, "config") and hasattr(model, "model") and hasattr(model, "dictionary")
        assert model.retriever == retriever
        assert model.config.model_name_or_path == MODELS[0]
        assert model.config.max_generation_length == 100

    def test_first_step_prompt(self, model: CHATEL) -> None:
        sentence = "This is a mention in the text about entity A."
        mention = "entity A"
        prompt = model.first_step_prompt(sentence, sentence.find(mention), sentence.find(mention) + len(mention))
        assert prompt == "This is a mention in the text about entity A. \n What does entity A in this sentence referring to?"

    def test_second_step_prompt(self, model: CHATEL) -> None:
        first_result = "Entity A refers to ..."
        mention = "entity A"
        candidates = [model.dictionary(c) for c in candidates_ids[0]]
        multi_choice_prompt = ""
        for i, cand in enumerate(candidates):
            description = cand['name'] + ' ' + cand['description'][:200] if cand['description'] else cand['name']
            multi_choice_prompt += f"({i+1}). {description}\n"

        second_prompt = model.second_step_prompt(first_result, mention, candidates_ids[0])
        assert second_prompt == f"Entity A refers to ...\n\nWhich of the following does {mention} referes to?\n\n{multi_choice_prompt}"

    def test_predict(self) -> None:
        sentence = "Apple is looking at buying U.K. startup for $1 billion"
        spans = [(0,5), (27,30), (44,52)]
        candidates = retriever.predict(sentence, spans, top_k=3)
        second_results = [
            "(1)",
            "(2)",
            "(3)"
        ]

        all_result = []
        for i, result in enumerate(second_results):
            assert isinstance(candidates[i], list) and len(candidates[i]) == 3
            assert isinstance(candidates[i][0], BaseSystemOutput)
            pred = process_multi_choice_prompt(result, [c.id for c in candidates[i]])
            all_result.append([candidates[i][pred]])
        assert len(all_result) == 3
        assert len(all_result[0]) == 1
        assert all_result[0][0].id == "000012"

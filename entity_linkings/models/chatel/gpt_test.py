from unittest.mock import MagicMock, patch

import pytest

from .gpt import OpenAI_API


class TestOpenAI_API:
    @pytest.fixture
    def api(self) -> OpenAI_API:
        return OpenAI_API(model_name="gpt-3.5-turbo", token="dummy-key")

    def test_generate(self, api: OpenAI_API) -> None:
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "hello, world!"

        with patch.object(api.client.chat.completions, 'create', return_value=mock_response) as mock_create:
            result = api._generate("hello")
            assert result == "hello, world!"

            mock_create.assert_called_once_with(
                model="gpt-3.5-turbo",
                messages=[ {"role": "user", "content": "hello"} ],
                temperature=0.2,
                top_p=0.0,
                max_tokens=4096,
                n=1
            )

        with patch.object(api.client.chat.completions, 'create', side_effect=Exception("API Connection Error")):
            result = api._generate("hello")
            assert "This process was terminated by the OpenAI API" in result

    def test_estimate(self, api: OpenAI_API) -> None:
        prompts = ["Hello, how are you?", "What is the capital of France?"]
        cost = api.estimate(prompts, estimated_output="Fine.")
        assert isinstance(cost, float)

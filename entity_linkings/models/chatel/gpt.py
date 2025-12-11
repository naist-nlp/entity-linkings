from typing import Optional

import tiktoken
from openai import OpenAI
from tqdm.auto import tqdm

from .utils import DEFAULT_TIMEOUT, Cache


class OpenAI_API:
    """OpenAI_APIWrapper

    This wrapper is a inclusive OpenAI API tools.
    I strongly recommend to debug with estimate() at first, then use BatchAPI(ex submit_batch()) in experiment.
    """
    def __init__(
            self,
            model_name: str,
            token: str,
            organization_key: Optional[str] = None,
            max_token_length: int = 4096,
            temperature: float = 0.2,
            top_p: float = 0.0,
            seed: int = 42,
            cache_dir: str = ".openai_cache",
        ) -> None:
        """__init__

        Args:
            model_name (str): OpenAI's model name
            api_file (str): Path of file of OpenAI api token. I assume that the api_token of OpenAI is formatting as json.
            total_cost_limit (float): Limit of total cost. This API terminates the process if the output cost is exceed of the limit. (Currency: USD)
            timeout (Optional[httpx.Timeout]): Timeout setting (Default: None)
        """
        self.client = OpenAI(organization=organization_key, api_key=token, timeout=DEFAULT_TIMEOUT)
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.max_tokens = max_token_length
        self.cache = Cache(model_name=model_name, cache_dir=cache_dir)

    def _generate(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[ {"role": "user", "content": prompt} ],
                temperature=self.temperature,
                top_p=self.top_p,
                n=1,
                max_tokens=self.max_tokens
            )
            generated_text = response.choices[0].message.content
            if not generated_text:
                raise ValueError("No response from OpenAI API.")
            return generated_text
        except Exception as e:
            return f"This process was terminated by the OpenAI API due to {str(e)}"

    def generate(self, prompts: str|list[str]) -> list[str]:
        generated_texts = []
        if isinstance(prompts, str):
            prompts = [prompts]
        pbar = tqdm(total=len(prompts), desc="Generating with OpenAI API")
        for prompt in prompts:
            pbar.update()
            if not self.cache.check_in_cache(prompt):
                _results = self._generate(prompt)
                results = self.cache.serialize(_results)
                self.cache.append_to_jsonl(prompt, results)
            else:
                results = self.cache(prompt)
            generated_texts.append(results)
        pbar.close()
        return generated_texts

    def estimate(self, prompts: str|list[str], estimated_output: str|list[str]) -> float:
        """Estimate cost of prompt

        Args:
            prompts (str|list[str]): prompt string or list of prompt strings

        Returns:
            float: estimated cost (Currency: USD)
        """
        tokenizer = tiktoken.encoding_for_model(self.model_name)
        prompt_tokens, generated_tokens = [], []
        if isinstance(prompts, str):
            assert isinstance(estimated_output, str)
            prompts = [prompts]
        if isinstance(estimated_output, str):
            estimated_outputs = [estimated_output]
        for prompt in prompts:
            prompt_tokens.extend(tokenizer.encode(prompt))
        for output in estimated_outputs:
            generated_tokens.extend(tokenizer.encode(output))
        cost = self._calculate_cost(self.model_name, len(prompt_tokens), len(generated_tokens) * len(prompts))
        return cost

    @staticmethod
    def _calculate_cost(model_name: str, num_prompt_tokens: int, num_generated_tokens: int = 0) -> float:
        cost_per_1k_tokens = {
            "gpt-3.5-turbo": {"prompt_tokens": 0.0005, "generated_tokens": 0.0015},
            "gpt-4.1": {"prompt_tokens": 0.002, "generated_tokens": 0.008},
            "gpt-4.1-mini": {"prompt_tokens": 0.0004, "generated_tokens": 0.0016},
            "gpt-4.1-nano": {"prompt_tokens": 0.0001, "generated_tokens": 0.0004},
            "gpt-4o": {"prompt_tokens": 0.0025, "generated_tokens": 0.01},
            "gpt-4o-mini": {"prompt_tokens": 0.00015, "generated_tokens": 0.0006}
        }
        if model_name not in cost_per_1k_tokens:
            raise ValueError(f"Cost for model {model_name} is not defined.")
        cost = 0.0
        cost += (num_prompt_tokens / 1000) * cost_per_1k_tokens[model_name]["prompt_tokens"]
        cost += (num_generated_tokens / 1000) * cost_per_1k_tokens[model_name]["generated_tokens"]
        return cost

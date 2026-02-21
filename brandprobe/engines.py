from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from openai import OpenAI, AzureOpenAI

class BaseEngine(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 250, temperature: float = 0.7) -> str:
        """Generate response given a system and user prompt."""
        pass

class AzureOpenAIEngine(BaseEngine):
    def __init__(self, api_key: str, api_version: str, azure_endpoint: str, deployment_name: str):
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint
        )
        self.deployment_name = deployment_name

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 250, temperature: float = 0.7) -> str:
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        if hasattr(response.choices[0].message, "content") and response.choices[0].message.content:
            return response.choices[0].message.content
        return ""

class OpenAIEngine(BaseEngine):
    def __init__(self, api_key: str = "ollama", base_url: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        
    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 250, temperature: float = 0.7) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        if hasattr(response.choices[0].message, "content") and response.choices[0].message.content:
            return response.choices[0].message.content
        return ""

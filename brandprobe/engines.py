from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from openai import OpenAI, AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

class BaseEngine(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 250, temperature: float = 0.7) -> str:
        """Generate response given a system and user prompt."""
        pass

class AzureOpenAIEngine(BaseEngine):
    def __init__(
        self,
        api_version: str,
        azure_endpoint: str,
        deployment_name: str,
        auth_mode: str = "entra",
        api_key: Optional[str] = None,
    ):
        self.deployment_name = deployment_name

        if auth_mode == "entra":
            token_provider = get_bearer_token_provider(
                DefaultAzureCredential(),
                "https://cognitiveservices.azure.com/.default",
            )
            self.client = AzureOpenAI(
                api_version=api_version,
                azure_endpoint=azure_endpoint,
                azure_ad_token_provider=token_provider,
            )
        elif auth_mode == "api_key":
            if not api_key:
                raise ValueError("api_key is required when auth_mode='api_key'")
            self.client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=azure_endpoint,
            )
        else:
            raise ValueError("auth_mode must be 'entra' or 'api_key'")

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

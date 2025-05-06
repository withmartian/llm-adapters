from typing import Any

from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.types import (
    Cost,
    Model,
    Provider,
    Vendor,
)
from openai.types.chat import ChatCompletionMessageParam


class TogetherModel(Model):
    provider_name: str = Provider.together.value

    supports_n: bool = False
    supports_completion: bool = False

    can_min_p: bool = False
    can_top_k: bool = False

    def _get_api_path(self) -> str:
        return f"{self.vendor_name}/{self.name}"


MODELS: list[Model] = [
    TogetherModel(
        name="DeepSeek-R1",
        vendor_name=Vendor.deepseek_ai.value,
        cost=Cost(prompt=3.00e-6, completion=7.00e-6),
        context_length=164000,
    ),
    TogetherModel(
        name="DeepSeek-V3",
        vendor_name=Vendor.deepseek_ai.value,
        cost=Cost(prompt=1.25e-6, completion=1.25e-6),
        context_length=131000,
        can_assistant_only=False,
    ),
    TogetherModel(
        name="Qwen2.5-72B-Instruct-Turbo",
        vendor_name=Vendor.Qwen.value,
        cost=Cost(prompt=1.20e-6, completion=1.20e-6),
        context_length=32768,
    ),
    TogetherModel(
        name="Mistral-Small-24B-Instruct-2501",
        vendor_name=Vendor.mistralai.value,
        cost=Cost(prompt=0.80e-6, completion=0.80e-6),
        context_length=32768,
        can_assistant_only=False,
    ),
    TogetherModel(
        name="Llama-3.1-Nemotron-70B-Instruct-HF",
        vendor_name=Vendor.nvidia.value,
        cost=Cost(prompt=0.88e-6, completion=0.88e-6),
        context_length=128000,
    ),
    TogetherModel(
        name="Llama-3.3-70B-Instruct-Turbo",
        vendor_name=Vendor.meta_llama.value,
        cost=Cost(prompt=0.88e-6, completion=0.88e-6),
        context_length=128000,
        supports_json_content=False,
    ),
    TogetherModel(
        name="Meta-Llama-3.1-405B-Instruct-Turbo",
        vendor_name=Vendor.meta_llama.value,
        cost=Cost(prompt=3.50e-6, completion=3.50e-6),
        context_length=131000,
        supports_json_content=False,
    ),
    TogetherModel(
        name="Qwen2.5-Coder-32B-Instruct",
        vendor_name=Vendor.Qwen.value,
        cost=Cost(prompt=0.80e-6, completion=0.80e-6),
        context_length=32768,
    ),
    TogetherModel(
        name="gemma-2-27b-it",
        vendor_name=Vendor.google.value,
        cost=Cost(prompt=0.80e-6, completion=0.80e-6),
        context_length=8000,
    ),
    # TogetherModel(
    #     name="DeepSeek-R1-Distill-Llama-70B",
    #     vendor_name=Vendor.deepseek_ai.value,
    #     cost=Cost(prompt=2.00e-6, completion=2.00e-6),
    #     context_length=128000,
    # ),
    # TogetherModel(
    #     name="DeepSeek-R1-Distill-Qwen-14",
    #     vendor_name=Vendor.deepseek_ai.value,
    #     cost=Cost(prompt=1.60e-6, completion=1.60e-6),
    #     context_length=128000,
    # ),
    TogetherModel(
        name="DeepSeek-R1-Distill-Qwen-1.5B",
        vendor_name=Vendor.deepseek_ai.value,
        cost=Cost(prompt=0.18e-6, completion=0.18e-6),
        context_length=128000,
    ),
    TogetherModel(
        name="Llama-4-Maverick-17B-128E-Instruct-FP8",
        vendor_name=Vendor.meta_llama.value,
        cost=Cost(prompt=0.27e-6, completion=0.85e-6),
        context_length=500000,
        supports_vision=True,
        can_empty_content=False,
    ),
    TogetherModel(
        name="Llama-4-Scout-17B-16E-Instruct",
        vendor_name=Vendor.meta_llama.value,
        cost=Cost(prompt=0.18e-6, completion=0.59e-6),
        context_length=300000,
        supports_vision=True,
        can_empty_content=False,
    ),
    # TogetherModel(
    #     name="Llama-3.3-70B-Instruct-Turbo-Free",
    #     vendor_name=Vendor.meta_llama.value,
    #     cost=Cost(prompt=0, completion=0),  # Free model
    #     context_length=8000,
    #     supports_json_content=False,
    # ),
    # TogetherModel(
    #     name="Llama-Vision-Free",
    #     vendor_name=Vendor.meta_llama.value,
    #     cost=Cost(prompt=0, completion=0),  # Free model
    #     context_length=128000,
    #     supports_vision=True,
    #     supports_json_output=False,
    #     can_empty_content=False,
    # ),
    TogetherModel(
        name="Llama-3.2-11B-Vision-Instruct-Turbo",
        vendor_name=Vendor.meta_llama.value,
        cost=Cost(prompt=0.18e-6, completion=0.18e-6),
        context_length=128000,
        supports_vision=True,
        supports_json_output=False,
        can_empty_content=False,
    ),
    TogetherModel(
        name="Llama-3.2-90B-Vision-Instruct-Turbo",
        vendor_name=Vendor.meta_llama.value,
        cost=Cost(prompt=1.20e-6, completion=1.20e-6),
        context_length=128000,
        supports_vision=True,
    ),
]

DEFAULT_TEMPERATURE = 0.7


class TogetherSDKChatProviderAdapter(OpenAISDKChatAdapter):
    @staticmethod
    def get_supported_models() -> list[Model]:
        return MODELS

    @staticmethod
    def get_api_key_name() -> str:
        return "TOGETHER_API_KEY"

    def get_base_sdk_url(self) -> str:
        return "https://api.together.xyz"

    def _adjust_temperature(self, temperature: float) -> float:
        return temperature / 2

    def _get_params(
        self, messages: list[ChatCompletionMessageParam], **kwargs: Any
    ) -> dict[str, Any]:
        params = super()._get_params(messages, **kwargs)

        # If the user has requested n messages, but not specified a temperature, we need to provide default temperature
        if params.get("n") and params.get("temperature") is None:
            params["temperature"] = DEFAULT_TEMPERATURE

        # Keep only last image_url for vision
        skiped_image = False
        for message in reversed(params["messages"]):
            if isinstance(message["content"], list):
                for content in reversed(message["content"]):
                    if content["type"] == "image_url":
                        if skiped_image:
                            content["type"] = "text"
                            content["text"] = content["image_url"]["url"]
                            del content["image_url"]
                        else:
                            skiped_image = True

        return params

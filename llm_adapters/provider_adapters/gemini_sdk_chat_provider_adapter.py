from llm_adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from llm_adapters.types import (
    Cost,
    Model,
    ModelProperties,
    Provider,
    Vendor,
)


class GeminiModel(Model):
    provider_name: str = Provider.gemini.value
    vendor_name: str = Vendor.gemini.value

    supports_completion: bool = False
    supports_tools: bool = False

    can_empty_content: bool = False
    can_system_last: bool = False

    properties: ModelProperties = ModelProperties(gdpr_compliant=True)


# TODO: setup pricing for longer contexts
MODELS: list[Model] = [
    GeminiModel(
        name="gemini-1.5-pro",
        cost=Cost(prompt=1.25e-6, completion=5.00e-6),
        context_length=2097152,
        completion_length=8192,
    ),
    GeminiModel(
        name="gemini-1.5-flash-8b",
        cost=Cost(prompt=0.0375e-6, completion=0.15e-6),
        context_length=1048576,
        completion_length=8192,
    ),
    GeminiModel(
        name="gemini-1.5-flash",
        cost=Cost(prompt=0.075e-6, completion=0.30e-6),
        context_length=1048576,
        completion_length=8192,
    ),
    GeminiModel(
        name="gemini-2.0-flash",
        cost=Cost(prompt=0.10e-6, completion=0.40e-6),
        context_length=1048576,
        completion_length=8192,
    ),
    GeminiModel(
        name="gemini-2.0-flash-lite",
        cost=Cost(prompt=0.075e-6, completion=0.30e-6),
        context_length=1048576,
        completion_length=8192,
        can_assistant_last=False,
    ),
    GeminiModel(
        name="gemini-2.0-flash",
        cost=Cost(prompt=0.0375e-6, completion=0.15e-6),
        context_length=1048576,
        completion_length=8192,
    ),
]


class GeminiSDKChatProviderAdapter(OpenAISDKChatAdapter):
    @staticmethod
    def get_supported_models() -> list[Model]:
        return MODELS

    @staticmethod
    def get_api_key_name() -> str:
        return "GEMINI_API_KEY"

    def get_base_sdk_url(self) -> str:
        return "https://generativelanguage.googleapis.com/v1beta/openai"

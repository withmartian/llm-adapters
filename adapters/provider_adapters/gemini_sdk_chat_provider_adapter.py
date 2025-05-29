from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.types import (
    Cost,
    Model,
    ModelProperties,
    Provider,
    Vendor,
)


class GeminiModel(Model):
    provider_name: str = Provider.gemini.value
    vendor_name: str = Vendor.gemini.value

    can_empty_content: bool = False

    supports_completion: bool = False

    properties: ModelProperties = ModelProperties(gdpr_compliant=True)


MODELS: list[Model] = [
    GeminiModel(
        name="gemini-1.5-pro-latest",
        cost=Cost(prompt=1.25e-6, completion=5.00e-6),
        context_length=2097152,
        completion_length=8192,
    ),
    GeminiModel(
        name="gemini-1.5-pro",
        cost=Cost(prompt=1.25e-6, completion=5.00e-6),
        context_length=2097152,
        completion_length=8192,
    ),
    GeminiModel(
        name="gemini-1.5-flash-latest",
        cost=Cost(prompt=0.075e-6, completion=0.30e-6),
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
        name="gemini-1.5-flash-8b-latest",
        cost=Cost(prompt=0.0375e-6, completion=0.15e-6),
        context_length=1048576,
        completion_length=8192,
    ),
    GeminiModel(
        name="gemini-1.5-flash-8b",
        cost=Cost(prompt=0.0375e-6, completion=0.15e-6),
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
        name="gemini-2.5-pro-preview-03-25",
        cost=Cost(prompt=0.15e-6, completion=0.6e-6),
        context_length=1048576,
        completion_length=65536,
        can_assistant_only=False,
    ),
    GeminiModel(
        name="gemini-2.5-flash-preview-04-17",
        cost=Cost(prompt=1.25e-6, completion=10.0e-6),
        context_length=1048576,
        completion_length=65536,
        can_assistant_only=False,
    ),
    GeminiModel(
        name="gemini-2.5-pro-preview-05-06",
        cost=Cost(prompt=1.25e-6, completion=10.0e-6),
        context_length=1048576,
        completion_length=65536,
        can_assistant_only=False,
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

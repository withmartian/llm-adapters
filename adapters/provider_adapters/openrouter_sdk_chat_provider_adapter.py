from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.types import Cost, Model, Provider, Vendor


class OpenRouterModel(Model):
    provider_name: str = Provider.openrouter.value
    
    supports_n: bool = False
    
    can_assistant_only: bool = False

    def _get_api_path(self) -> str:
        return f"{self.vendor_name}/{self.name}"


MODELS: list[Model] = [
    OpenRouterModel(
        name="devstral-small",
        vendor_name=Vendor.mistralai.value,
        cost=Cost(prompt=0.07e-6, completion=0.28e-6),
        context_length=128000,
    ),
    OpenRouterModel(
        name="devstral-medium",
        vendor_name=Vendor.mistralai.value,
        cost=Cost(prompt=0.40e-6, completion=2.08e-6),
        context_length=131072,
    ),
    OpenRouterModel(
        name="codestral-2501",
        vendor_name=Vendor.mistralai.value,
        cost=Cost(prompt=0.30e-6, completion=0.90e-6),
        context_length=262144,
    ),
    OpenRouterModel(
        name="kimi-k2",
        vendor_name=Vendor.moonshotai.value,
        cost=Cost(prompt=0.14e-6, completion=2.49e-6),
        context_length=63000,
    ),
]


class OpenRouterSDKChatProviderAdapter(OpenAISDKChatAdapter):
    @staticmethod
    def get_supported_models() -> list[Model]:
        return MODELS

    @staticmethod
    def get_api_key_name() -> str:
        return "OPENROUTER_API_KEY"

    def get_base_sdk_url(self) -> str:
        return "https://openrouter.ai/api/v1"

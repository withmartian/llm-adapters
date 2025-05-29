from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.types import Cost, Model, Provider, Vendor


class PerplexityModel(Model):
    provider_name: str = Provider.perplexity.value

    supports_completion: bool = False
    supports_last_system: bool = False
    supports_json_output: bool = False
    supports_tools: bool = False
    supports_n: bool = False
    supports_stop: bool = False

    can_assistant_first: bool = False
    can_assistant_last: bool = False
    can_assistant_only: bool = False
    can_developer: bool = False

    can_system_last: bool = False

    can_empty_content: bool = False
    can_repeating_roles: bool = False
    can_system_multiple: bool = False
    can_min_p: bool = False
    can_top_k: bool = False


MODELS: list[Model] = [
    PerplexityModel(
        name="sonar",
        cost=Cost(prompt=1.00e-6, completion=1.00e-6, request=0.01),
        context_length=127000,
        vendor_name=Vendor.perplexity.value,
    ),
    PerplexityModel(
        name="sonar-pro",
        cost=Cost(prompt=3.00e-6, completion=15.00e-6, request=0.01),
        context_length=127000,
        completion_length=8000,
        vendor_name=Vendor.perplexity.value,
    ),
    PerplexityModel(
        name="sonar-reasoning",
        cost=Cost(prompt=1.00e-6, completion=5.00e-6, request=0.01),
        context_length=127000,
        vendor_name=Vendor.perplexity.value,
    ),
    PerplexityModel(
        name="sonar-reasoning-pro",
        cost=Cost(prompt=2.00e-6, completion=8.00e-6, request=0.01),
        context_length=127000,
        vendor_name=Vendor.perplexity.value,
    ),
]


class PerplexitySDKChatProviderAdapter(OpenAISDKChatAdapter):
    @staticmethod
    def get_supported_models() -> list[Model]:
        return MODELS

    @staticmethod
    def get_api_key_name() -> str:
        return "PERPLEXITY_API_KEY"

    def get_base_sdk_url(self) -> str:
        return "https://api.perplexity.ai"

from llm_adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from llm_adapters.types import Model, Provider, Vendor


# An average token corresponds to 1 word or 6 characters of English text.
# Comparatively, this is up to 30% more text per token than other providers, saving you 30% of the cost.
class AI21Model(Model):
    provider_name: str = Provider.ai21.value
    vendor_name: str = Vendor.ai21.value

    supports_tools: bool = False
    supports_completion: bool = False
    supports_json_content: bool = False
    supports_vision: bool = False
    supports_streaming: bool = False

    can_system_only: bool = False
    can_empty_content: bool = False
    can_min_p: bool = False
    can_top_k: bool = False


MODELS: list[Model] = [
    # AI21Model(
    #     name="jamba-1.6-mini",
    #     cost=Cost(prompt=0.2e-6, completion=0.4e-6),
    #     context_length=256000,
    # ),
    # AI21Model(
    #     name="jamba-1.6-large",
    #     cost=Cost(prompt=2.0e-6, completion=8.0e-6),
    #     context_length=256000,
    # ),
]


class AI21SDKChatProviderAdapter(OpenAISDKChatAdapter):
    @staticmethod
    def get_supported_models() -> list[Model]:
        return MODELS

    @staticmethod
    def get_api_key_name() -> str:
        return "AI21_API_KEY"

    def get_base_sdk_url(self) -> str:
        return "https://api.ai21.com/studio/v1"

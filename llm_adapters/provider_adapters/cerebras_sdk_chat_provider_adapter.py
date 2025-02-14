from llm_adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from llm_adapters.types import Cost, Model, Provider, Vendor


class CerebrasModel(Model):
    provider_name: str = Provider.cerebras.value

    supports_n: bool = False
    supports_vision: bool = False

    can_min_p: bool = False
    can_top_k: bool = False


MODELS: list[Model] = [
    CerebrasModel(
        name="llama3.1-8b",
        vendor_name=Vendor.meta_llama.value,
        cost=Cost(prompt=0.10e-6, completion=0.10e-6),
        context_length=32768,
    ),
    CerebrasModel(
        name="llama3.3-70b",
        vendor_name=Vendor.meta_llama.value,
        cost=Cost(prompt=0.85e-6, completion=1.20e-6),
        context_length=32768,
    ),
    # CerebrasModel(
    #     name="deepseek-r1-distill-llama-70b",
    #     vendor_name=Vendor.deepseek.value,
    #     cost=Cost(prompt=0.85e-6, completion=1.20e-6),
    #     context_length=65536,
    # ),
]


class CerebrasSDKChatProviderAdapter(OpenAISDKChatAdapter):
    @staticmethod
    def get_supported_models() -> list[Model]:
        return MODELS

    @staticmethod
    def get_api_key_name() -> str:
        return "CEREBRAS_API_KEY"

    def get_base_sdk_url(self) -> str:
        return "https://api.cerebras.ai/v1"

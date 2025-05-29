from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.types import Cost, Model, Provider, Vendor


class CerebrasModel(Model):
    provider_name: str = Provider.cerebras.value

    supports_max_completion_tokens: bool = False

    supports_n: bool = False

    can_min_p: bool = False
    can_top_k: bool = False

    def _get_api_path(self) -> str:
        return f"{self.name}"


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
    #     name="llama-4-scout-17b-16e-instruct",
    #     vendor_name=Vendor.meta_llama.value,
    #     cost=Cost(prompt=0.65e-6, completion=0.85e-6),
    #     context_length=32768,
    # ),
    # CerebrasModel(
    #     name="qwen-3-32b",
    #     vendor_name=Vendor.meta_llama.value,
    #     cost=Cost(prompt=0.40e-6, completion=0.80e-6),
    #     context_length=32768,
    # ),
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

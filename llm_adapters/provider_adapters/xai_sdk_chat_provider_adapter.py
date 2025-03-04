from llm_adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from llm_adapters.types import Cost, Model, Provider, Vendor


class XAIModel(Model):
    provider_name: str = Provider.xai.value
    vendor_name: str = Vendor.grok.value


MODELS: list[Model] = [
    XAIModel(
        name="grok-2",
        cost=Cost(prompt=2.00e-6, completion=10.00e-6),
        context_length=131072,
    ),
    # TODO: Add vision pricing
    XAIModel(
        name="grok-2-vision",
        cost=Cost(prompt=2.00e-6, completion=10.00e-6),
        context_length=32768,
    ),
]


class XAISDKChatProviderAdapter(OpenAISDKChatAdapter):
    @staticmethod
    def get_supported_models() -> list[Model]:
        return MODELS

    @staticmethod
    def get_api_key_name() -> str:
        return "XAI_API_KEY"

    def get_base_sdk_url(self) -> str:
        return "https://api.x.ai/v1"

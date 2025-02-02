from llm_adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from llm_adapters.types import Cost, Model, Provider, Vendor


class OpenAIModel(Model):
    provider_name: str = Provider.openai.value
    vendor_name: str = Vendor.openai.value

    supports_completion: bool = False
    can_min_p: bool = False
    can_top_k: bool = False


MODELS: list[Model] = [
    OpenAIModel(
        name="gpt-3.5-turbo-instruct",
        cost=Cost(prompt=1.50e-6, completion=2.00e-6),
        context_length=16385,
        completion_length=16385,
        supports_completion=True,
        supports_chat=False,
        supports_max_completion_tokens=False,
    ),
    OpenAIModel(
        name="gpt-4o",
        cost=Cost(prompt=2.50e-6, completion=10.00e-6),
        context_length=128000,
        completion_length=16384,
    ),
    OpenAIModel(
        name="gpt-4o-2024-11-20",
        cost=Cost(prompt=2.50e-6, completion=10.00e-6),
        context_length=128000,
        completion_length=16384,
    ),
    OpenAIModel(
        name="gpt-4o-mini",
        cost=Cost(prompt=0.15e-6, completion=0.60e-6),
        context_length=128000,
        completion_length=16385,
    ),
    OpenAIModel(
        name="o1-mini",
        cost=Cost(prompt=3.00e-6, completion=12.00e-6),
        context_length=128000,
        completion_length=65536,
        supports_tools=False,
        supports_streaming=False,
        supports_n=False,
        supports_vision=False,
        supports_json_output=False,
        supports_max_tokens=False,
        supports_stop=False,
        can_temperature=False,
        can_system=False,
        can_top_p=False,
    ),
    OpenAIModel(
        name="o1",
        cost=Cost(prompt=15.00e-6, completion=60.00e-6),
        context_length=200000,
        completion_length=100000,
        supports_tools=False,
        supports_streaming=False,
        supports_n=False,
        supports_vision=False,
        supports_json_output=False,
        supports_max_tokens=False,
        supports_stop=False,
        can_temperature=False,
        can_system=False,
        can_top_p=False,
    ),
]


class OpenAISDKChatProviderAdapter(OpenAISDKChatAdapter):
    @staticmethod
    def get_supported_models() -> list[Model]:
        return MODELS

    @staticmethod
    def get_api_key_name() -> str:
        return "OPENAI_API_KEY"

    def get_base_sdk_url(self) -> str:
        return "https://api.openai.com/v1"

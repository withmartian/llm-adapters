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
        name="gpt-4o-2024-08-06",
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
        name="gpt-4o-mini-2024-07-18",
        cost=Cost(prompt=0.15e-6, completion=0.60e-6),
        context_length=128000,
        completion_length=16385,
    ),
    OpenAIModel(
        name="o1-mini",
        cost=Cost(prompt=1.10e-6, completion=4.40e-6),
        context_length=128000,
        completion_length=65536,
        supports_json_output=False,
        supports_stop=False,
        supports_tools=False,
    ),
    OpenAIModel(
        name="o1-mini-2024-09-12",
        cost=Cost(prompt=1.10e-6, completion=4.40e-6),
        context_length=128000,
        completion_length=65536,
        supports_json_output=False,
        supports_stop=False,
        supports_tools=False,
    ),
    OpenAIModel(
        name="o1",
        cost=Cost(prompt=15.00e-6, completion=60.00e-6),
        context_length=200000,
        completion_length=100000,
    ),
    OpenAIModel(
        name="o1-2024-12-17",
        cost=Cost(prompt=15.00e-6, completion=60.00e-6),
        context_length=200000,
        completion_length=100000,
    ),
    OpenAIModel(
        name="o3-mini",
        cost=Cost(prompt=1.10e-6, completion=4.40e-6),
        context_length=200000,
        completion_length=100000,
    ),
    OpenAIModel(
        name="o3-mini-2025-01-31",
        cost=Cost(prompt=1.10e-6, completion=4.40e-6),
        context_length=200000,
        completion_length=100000,
    ),
    OpenAIModel(
        name="gpt-4.5-preview-2025-02-27",
        cost=Cost(prompt=75.00e-6, completion=150.00e-6),
        context_length=128000,
        completion_length=16384,
        supports_n=False,
        supports_stop=False,
        can_system_last=False,
        can_repeating_roles=False,
        can_empty_content=False,
        can_temperature=False,
    ),
    OpenAIModel(
        name="gpt-4.5-preview",
        cost=Cost(prompt=75.00e-6, completion=150.00e-6),
        context_length=128000,
        completion_length=16384,
        supports_n=False,
        supports_stop=False,
        can_system_last=False,
        can_repeating_roles=False,
        can_empty_content=False,
        can_temperature=False,
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

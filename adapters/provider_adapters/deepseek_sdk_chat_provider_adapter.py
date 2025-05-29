from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.types import Cost, Model, Provider, Vendor


class DeepSeekModel(Model):
    provider_name: str = Provider.deepseek.value
    supports_vision: bool = False
    vendor_name: str = Vendor.deepseek.value
    supports_n: bool = False

    def _get_api_path(self) -> str:
        return f"{self.name}"


MODELS: list[Model] = [
    DeepSeekModel(
        name="deepseek-chat",
        cost=Cost(prompt=0.27e-6, completion=1.10e-6),
        context_length=64000,
        completion_length=8000,
        supports_completion=False,  # Available in beta mode
    ),
    DeepSeekModel(
        name="deepseek-reasoner",
        cost=Cost(prompt=0.55e-6, completion=2.19e-6),
        context_length=64000,
        supports_json_output=False,
        supports_completion=False,
        supports_max_tokens=False,
        supports_tools=False,
        can_assistant_first=False,
        can_assistant_last=False,
        can_assistant_only=False,
        can_repeating_roles=False,
        can_system_last=False,
        can_system_only=False,
        can_system_multiple=False,
    ),
]


class DeepSeekSDKChatProviderAdapter(OpenAISDKChatAdapter):
    @staticmethod
    def get_supported_models() -> list[Model]:
        return MODELS

    @staticmethod
    def get_api_key_name() -> str:
        return "DEEPSEEK_API_KEY"

    def get_base_sdk_url(self) -> str:
        return "https://api.deepseek.com"

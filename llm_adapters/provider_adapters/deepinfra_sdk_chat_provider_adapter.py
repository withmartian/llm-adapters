from llm_adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from llm_adapters.types import Cost, Model, Provider, Vendor


class DeepInfraModel(Model):
    provider_name: str = Provider.deepinfra.value

    supports_completion: bool = False
    supports_vision: bool = False
    supports_tools: bool = False
    supports_json_content: bool = False
    supports_n: bool = False

    can_assistant_only: bool = False
    can_system_only: bool = False
    can_min_p: bool = False
    can_top_k: bool = False

    def get_api_path(self) -> str:
        return f"{self.vendor_name}/{self.name}"


# TODO: check completion length
# TODO: try lower case model names
# TODO: check json support
# TODO: check function/tools calling support
MODELS: list[Model] = [
    DeepInfraModel(
        name="Llama-3.3-70B-Instruct",
        cost=Cost(prompt=0.23e-6, completion=0.40 - 6),
        context_length=131072,
        vendor_name=Vendor.meta_llama,
    ),
    DeepInfraModel(
        name="Llama-3.3-70B-Instruct-Turbo",
        cost=Cost(prompt=0.12e-6, completion=0.30e-6),
        context_length=131072,
        vendor_name=Vendor.meta_llama,
    ),
    DeepInfraModel(
        name="phi-4",
        cost=Cost(prompt=0.07e-6, completion=0.14e-6),
        context_length=16384,
        vendor_name=Vendor.microsoft,
    ),
    DeepInfraModel(
        name="DeepSeek-V3",
        cost=Cost(prompt=0.85e-6, completion=0.90e-6),
        context_length=16000,
        vendor_name=Vendor.deepseek,
    ),
    DeepInfraModel(
        name="DeepSeek-R1",
        cost=Cost(prompt=0.85e-6, completion=2.50e-6),
        context_length=16000,
        vendor_name=Vendor.deepseek,
    ),
    DeepInfraModel(
        name="DeepSeek-R1-Distill-Llama-70B",
        cost=Cost(prompt=0.23e-6, completion=0.69e-6),
        context_length=131072,
        vendor_name=Vendor.deepseek,
    ),
    DeepInfraModel(
        name="QwQ-32B-Preview",
        cost=Cost(prompt=0.12e-6, completion=0.18e-6),
        context_length=32768,
        vendor_name=Vendor.Qwen,
    ),
    DeepInfraModel(
        name="Qwen2.5-Coder-32B-Instruct",
        cost=Cost(prompt=0.07e-6, completion=0.16e-6),
        context_length=32768,
        vendor_name=Vendor.Qwen,
    ),
    DeepInfraModel(
        name="Qwen2.5-72B-Instruct",
        cost=Cost(prompt=0.23e-6, completion=0.40e-6),
        context_length=32768,
        vendor_name=Vendor.Qwen,
    ),
    DeepInfraModel(
        name="WizardLM-2-8x22B",
        cost=Cost(prompt=0.50e-6, completion=0.50e-6),
        context_length=65536,
        vendor_name=Vendor.microsoft,
    ),
]


class DeepInfraSDKChatProviderAdapter(OpenAISDKChatAdapter):
    @staticmethod
    def get_supported_models() -> list[Model]:
        return MODELS

    @staticmethod
    def get_api_key_name() -> str:
        return "DEEPINFRA_API_KEY"

    def get_base_sdk_url(self) -> str:
        return "https://api.deepinfra.com/v1/openai"

    def _adjust_temperature(self, temperature: float) -> float:
        return temperature / 2

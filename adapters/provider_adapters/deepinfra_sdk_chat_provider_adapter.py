from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.types import Cost, Model, Provider, Vendor


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
    can_developer: bool = False

    def get_api_path(self) -> str:
        return f"{self.vendor_name}/{self.name}"


# TODO: check completion length
# TODO: try lower case model names
# TODO: check json support
# TODO: check function/tools calling support
MODELS: list[Model] = [
    DeepInfraModel(
        name="Llama-3.3-70B-Instruct",
        cost=Cost(prompt=0.23e-6, completion=0.40e-6),
        context_length=131072,
        vendor_name=Vendor.meta_llama,
    ),
    DeepInfraModel(
        name="Llama-3.3-70B-Instruct-Turbo",
        cost=Cost(prompt=0.07e-6, completion=0.25e-6),
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
        cost=Cost(prompt=0.38e-6, completion=0.89e-6),
        context_length=16000,
        vendor_name=Vendor.deepseek_ai.value,
        can_system_only=False,
    ),
    DeepInfraModel(
        name="DeepSeek-R1",
        cost=Cost(prompt=0.45e-6, completion=2.18e-6),
        context_length=16000,
        vendor_name=Vendor.deepseek_ai.value,
        can_system_only=False,
    ),
    DeepInfraModel(
        name="DeepSeek-R1-Distill-Llama-70B",
        cost=Cost(prompt=0.10e-6, completion=0.40e-6),
        context_length=131072,
        vendor_name=Vendor.deepseek_ai.value,
        can_system_only=False,
    ),
    # DeepInfraModel(
    #     name="QwQ-32B",
    #     cost=Cost(prompt=0.15e-6, completion=0.20e-6),
    #     context_length=131072,
    #     vendor_name=Vendor.Qwen,
    # ),
    DeepInfraModel(
        name="Qwen2.5-Coder-32B-Instruct",
        cost=Cost(prompt=0.06e-6, completion=0.15e-6),
        context_length=32768,
        vendor_name=Vendor.Qwen,
    ),
    DeepInfraModel(
        name="Qwen2.5-72B-Instruct",
        cost=Cost(prompt=0.12e-6, completion=0.39e-6),
        context_length=32768,
        vendor_name=Vendor.Qwen,
    ),
    DeepInfraModel(
        name="WizardLM-2-8x22B",
        cost=Cost(prompt=0.50e-6, completion=0.50e-6),
        context_length=65536,
        vendor_name=Vendor.microsoft,
    ),
    DeepInfraModel(
        name="Llama-4-Maverick-17B-128E-Instruct-FP8",
        cost=Cost(prompt=0.16e-6, completion=0.60e-6),
        context_length=131072,
        vendor_name=Vendor.meta_llama,
        can_empty_content=False,
    ),
    DeepInfraModel(
        name="Llama-4-Scout-17B-16E-Instruct",
        cost=Cost(prompt=0.08e-6, completion=0.30e-6),
        context_length=131072,
        vendor_name=Vendor.meta_llama,
        can_empty_content=False,
    ),
    DeepInfraModel(
        name="DeepSeek-R1-Turbo",
        cost=Cost(prompt=1.00e-6, completion=3.00e-6),
        context_length=32768,
        vendor_name=Vendor.deepseek_ai,
    ),
    DeepInfraModel(
        name="QwQ-32B",
        cost=Cost(prompt=0.15e-6, completion=0.20e-6),
        context_length=131072,
        vendor_name=Vendor.Qwen,
    ),
    DeepInfraModel(
        name="DeepSeek-V3-0324",
        cost=Cost(prompt=0.30e-6, completion=0.88e-6),
        context_length=163840,
        vendor_name=Vendor.deepseek_ai,
    ),
    DeepInfraModel(
        name="gemma-3-27b-it",
        cost=Cost(prompt=0.10e-6, completion=0.20e-6),
        context_length=131072,
        vendor_name=Vendor.google,
        can_empty_content=False,
    ),
    DeepInfraModel(
        name="gemma-3-12b-it",
        cost=Cost(prompt=0.05e-6, completion=0.10e-6),
        context_length=131072,
        vendor_name=Vendor.google,
        can_empty_content=False,
    ),
    DeepInfraModel(
        name="gemma-3-4b-it",
        cost=Cost(prompt=0.02e-6, completion=0.04e-6),
        context_length=131072,
        vendor_name=Vendor.google,
        can_empty_content=False,
    ),
    DeepInfraModel(
        name="Phi-4-multimodal-instruct",
        cost=Cost(prompt=0.05e-6, completion=0.10e-6),
        context_length=131072,
        vendor_name=Vendor.microsoft,
    ),
    # DeepInfraModel(
    #     name="DeepSeek-R1-Distill-Qwen-32B",
    #     cost=Cost(prompt=12e-6, completion=18e-6),
    #     context_length=131072,
    #     vendor_name=Vendor.deepseek_ai,
    # ),
    DeepInfraModel(
        name="Meta-Llama-3.1-405B-Instruct",
        cost=Cost(prompt=0.80e-6, completion=0.80e-6),
        context_length=32768,
        vendor_name=Vendor.meta_llama,
    ),
    DeepInfraModel(
        name="Meta-Llama-3.1-70B-Instruct",
        cost=Cost(prompt=0.23e-6, completion=0.40e-6),
        context_length=131072,
        vendor_name=Vendor.meta_llama,
    ),
    DeepInfraModel(
        name="Meta-Llama-3.1-8B-Instruct",
        cost=Cost(prompt=0.03e-6, completion=0.05e-6),
        context_length=131072,
        vendor_name=Vendor.meta_llama,
    ),
    DeepInfraModel(
        name="Meta-Llama-3.1-70B-Instruct-Turbo",
        cost=Cost(prompt=0.10e-6, completion=0.28e-6),
        context_length=131072,
        vendor_name=Vendor.meta_llama,
    ),
    DeepInfraModel(
        name="Meta-Llama-3.1-8B-Instruct-Turbo",
        cost=Cost(prompt=0.02e-6, completion=0.05e-6),
        context_length=131072,
        vendor_name=Vendor.meta_llama,
    ),
    DeepInfraModel(
        name="Mistral-Small-24B-Instruct-2501",
        cost=Cost(prompt=0.06e-6, completion=0.12e-6),
        context_length=32768,
        vendor_name=Vendor.mistralai,
    ),
    DeepInfraModel(
        name="Mixtral-8x7B-Instruct-v0.1",
        cost=Cost(prompt=0.08e-6, completion=0.24e-6),
        context_length=32768,
        vendor_name=Vendor.mistralai,
    ),
    DeepInfraModel(
        name="Llama-3.2-90B-Vision-Instruct",
        cost=Cost(prompt=0.35e-6, completion=0.40e-6),
        context_length=32768,
        vendor_name=Vendor.meta_llama,
    ),
    DeepInfraModel(
        name="Llama-3.2-11B-Vision-Instruct",
        cost=Cost(prompt=0.049e-6, completion=0.049e-6),
        context_length=131072,
        vendor_name=Vendor.meta_llama,
    ),
    DeepInfraModel(
        name="Llama-3.2-1B-Instruct",
        cost=Cost(prompt=0.005e-6, completion=0.01e-6),
        context_length=131072,
        vendor_name=Vendor.meta_llama,
    ),
    DeepInfraModel(
        name="Llama-3.2-3B-Instruct",
        cost=Cost(prompt=0.01e-6, completion=0.02e-6),
        context_length=131072,
        vendor_name=Vendor.meta_llama,
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

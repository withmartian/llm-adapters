from llm_adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from llm_adapters.types import Cost, Model, Provider, Vendor


class LambdaLabsModel(Model):
    provider_name: str = Provider.lambdalabs.value

    supports_completion: bool = False

    supports_n: bool = False
    supports_json_output: bool = False
    supports_stop: bool = False
    supports_vision: bool = False
    supports_tools: bool = False

    can_min_p: bool = False
    can_top_k: bool = False

    def _get_api_path(self) -> str:
        return f"{self.name}"


MODELS: list[Model] = [
    LambdaLabsModel(
        name="llama3.1-405b-instruct-fp8",
        vendor_name=Vendor.meta_llama.value,
        cost=Cost(prompt=0.90e-6, completion=0.90e-6),
        context_length=131000,
    ),
    LambdaLabsModel(
        name="lfm-40b",
        vendor_name=Vendor.liquid.value,
        cost=Cost(prompt=0.15e-6, completion=0.15e-6),
        context_length=66000,
        supports_max_completion_tokens=False,
    ),
    LambdaLabsModel(
        name="llama3.1-8b-instruct",
        vendor_name=Vendor.meta_llama.value,
        cost=Cost(prompt=0.03e-6, completion=0.03e-6),
        context_length=131000,
    ),
    # LambdaLabsModel(
    #     name="llama3.2-11b-vision-instruct",
    #     vendor_name=Vendor.meta_llama.value,
    #     cost=Cost(
    #         prompt=0.0e-6, completion=0.0e-6
    #     ),  # pricing not provided in the table
    #     context_length=131000,
    # ),
    LambdaLabsModel(
        name="qwen25-coder-32b-instruct",
        vendor_name=Vendor.qwen.value,
        cost=Cost(prompt=0.09e-6, completion=0.09e-6),
        context_length=33000,
    ),
    LambdaLabsModel(
        name="hermes3-405b",
        vendor_name=Vendor.nous_hermes.value,
        cost=Cost(prompt=0.90e-6, completion=0.90e-6),
        context_length=131000,
        can_assistant_only=False,
    ),
    LambdaLabsModel(
        name="deepseek-llama3.3-70b",
        vendor_name=Vendor.deepseek.value,
        cost=Cost(prompt=0.20e-6, completion=0.20e-6),
        context_length=131000,
    ),
    LambdaLabsModel(
        name="llama3.1-70b-instruct-fp8",
        vendor_name=Vendor.meta_llama.value,
        cost=Cost(prompt=0.20e-6, completion=0.20e-6),
        context_length=131000,
    ),
    LambdaLabsModel(
        name="llama3.3-70b-instruct-fp8",
        vendor_name=Vendor.meta_llama.value,
        cost=Cost(prompt=0.20e-6, completion=0.20e-6),
        context_length=131000,
    ),
    LambdaLabsModel(
        name="llama3.2-3b-instruct",
        vendor_name=Vendor.meta_llama.value,
        cost=Cost(prompt=0.02e-6, completion=0.02e-6),
        context_length=131000,
    ),
    LambdaLabsModel(
        name="hermes3-8b",
        vendor_name=Vendor.nous_hermes.value,
        cost=Cost(prompt=0.03e-6, completion=0.03e-6),
        context_length=131000,
    ),
    LambdaLabsModel(
        name="llama3.1-nemotron-70b-instruct-fp8",
        vendor_name=Vendor.meta_llama.value,
        cost=Cost(prompt=0.20e-6, completion=0.20e-6),
        context_length=131000,
    ),
    LambdaLabsModel(
        name="hermes3-70b",
        vendor_name=Vendor.nous_hermes.value,
        cost=Cost(prompt=0.20e-6, completion=0.20e-6),
        context_length=131000,
        can_assistant_only=False,
    ),
]


class LambdaLabsSDKChatProviderAdapter(OpenAISDKChatAdapter):
    @staticmethod
    def get_supported_models() -> list[Model]:
        return MODELS

    @staticmethod
    def get_api_key_name() -> str:
        return "LAMBDALABS_API_KEY"

    def get_base_sdk_url(self) -> str:
        return "https://api.lambdalabs.com/v1"

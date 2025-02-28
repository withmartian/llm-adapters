from llm_adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from llm_adapters.types import Model


class CustomOpenAISDKChatProviderAdapter(OpenAISDKChatAdapter):
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        super().__init__()

    @staticmethod
    def get_supported_models() -> list[Model]:
        return []

    def get_base_sdk_url(self) -> str:
        return self.base_url

    @staticmethod
    def get_api_key_name() -> str:
        return ""


# Deprecated, use CustomOpenAISDKChatProviderAdapter instead
CustomAISDKChatProviderAdapter = CustomOpenAISDKChatProviderAdapter


class CustomOpenAIEnterpriseSDKChatProviderAdapter(OpenAISDKChatAdapter):
    def __init__(self, base_url: str) -> None:
        self.base_url = "custom_base_endpoint"
        super().__init__()

    @staticmethod
    def get_supported_models() -> list[Model]:
        return [
            Model(
                name="custom_model",
                provider_name="custom",
                vendor_name="custom",
                base_url="custom_base_endpoint",
            )
        ]

    def get_base_sdk_url(self) -> str:
        return self.base_url

    @staticmethod
    def get_api_key_name() -> str:
        return "custom_api_key"


AdapterFactory.register_adapter(CustomOpenAIEnterpriseSDKChatProviderAdapter)


adapterFactory.get_adapter(adapterFactory.get_model("custom/custom/custom_model"))
custom_adapter.execute()


# openai_client = OpenAI(
#     base_url="custom_base_endpoint",
#     api_key="custom_api_key",
# )

# openai_client.chat.completions.create(
#     model="custom_model",
#     messages=[{"role": "user", "content": "Hello, world!"}],
# )

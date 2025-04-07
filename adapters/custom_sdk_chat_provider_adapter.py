from llm_adapters.custom_sdk_chat_provider_adapter import (
    CustomOpenAISDKChatProviderAdapter,
)

# Deprecated, use CustomOpenAISDKChatProviderAdapter instead
CustomAISDKChatProviderAdapter = CustomOpenAISDKChatProviderAdapter

__all__ = ["CustomOpenAISDKChatProviderAdapter", "CustomAISDKChatProviderAdapter"]

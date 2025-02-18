# from typing import Any

# from llm_adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
# from llm_adapters.types import (
#     Cost,
#     Model,
#     Provider,
#     Vendor,
# )
# from openai.types.chat import ChatCompletionMessageParam


# class TogetherModel(Model):
#     provider_name: str = Provider.together.value

#     supports_completion: bool = False
#     can_min_p: bool = False
#     can_top_k: bool = False

#     def get_api_path(self) -> str:
#         return f"{self.vendor_name}/{self.name}"


# MODELS: list[Model] = [
# ]

# DEFAULT_TEMPERATURE = 0.7


# class TogetherSDKChatProviderAdapter(OpenAISDKChatAdapter):
#     @staticmethod
#     def get_supported_models() -> list[Model]:
#         return MODELS

#     @staticmethod
#     def get_api_key_name() -> str:
#         return "TOGETHER_API_KEY"

#     def get_base_sdk_url(self) -> str:
#         return "https://api.together.xyz"

#     def _adjust_temperature(self, temperature: float) -> float:
#         return temperature / 2

#     def _get_params(
#         self, messages: list[ChatCompletionMessageParam], **kwargs: Any
#     ) -> dict[str, Any]:
#         params = super()._get_params(messages, **kwargs)

#         # If the user has requested n messages, but not specified a temperature, we need to provide default temperature
#         if params.get("n") and params.get("temperature") is None:
#             params["temperature"] = DEFAULT_TEMPERATURE

#         # Keep only last image_url for vision
#         skiped_image = False
#         for message in reversed(params["messages"]):
#             if isinstance(message["content"], list):
#                 for content in reversed(message["content"]):
#                     if content["type"] == "image_url":
#                         if skiped_image:
#                             content["type"] = "text"
#                             content["text"] = content["image_url"]["url"]
#                             del content["image_url"]
#                         else:
#                             skiped_image = True

#         return params

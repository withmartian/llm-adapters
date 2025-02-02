# from llm_adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
# from llm_adapters.general_utils import YUAN_TO_USD
# from llm_adapters.types import (
#     Cost,
#     Model,
#     ModelProperties,
#     Provider,
#     Vendor,
# )


# class BigModelModel(Model):
#     provider_name: str = Provider.bigmodel.value
#     vendor_name: str = Vendor.bigmodel.value

#     supports_completion: bool = False
#     supports_vision: bool = False
#     supports_tools: bool = False
#     supports_tool_choice: bool = False
#     supports_tool_choice_required: bool = False
#     supports_n: bool = False
#     supports_stop: bool = False

#     can_empty_content: bool = False
#     can_system_only: bool = False
#     can_assistant_only: bool = False
#     can_min_p: bool = False
#     can_top_k: bool = False

#     properties: ModelProperties = ModelProperties(
#         gdpr_compliant=False,
#     )


# # TODO: check completion length
# MODELS: list[Model] = [
#     BigModelModel(
#         name="glm-4-plus",
#         cost=Cost(prompt=0.00005e-6 * YUAN_TO_USD, completion=0.00005e-6 * YUAN_TO_USD),
#         context_length=128000,
#         completion_length=4096,
#     ),
#     BigModelModel(
#         name="glm-4-air-0111",
#         cost=Cost(
#             prompt=0.0000005e-6 * YUAN_TO_USD, completion=0.0000005e-6 * YUAN_TO_USD
#         ),
#         context_length=128000,
#         completion_length=4096,
#     ),
#     BigModelModel(
#         name="glm-4-long",
#         cost=Cost(
#             prompt=0.000001e-6 * YUAN_TO_USD, completion=0.000001e-6 * YUAN_TO_USD
#         ),
#         context_length=1000000,
#         completion_length=4096,
#     ),
#     BigModelModel(
#         name="glm-zero-preview",
#         cost=Cost(prompt=0.00001e-6 * YUAN_TO_USD, completion=0.00001e-6 * YUAN_TO_USD),
#         context_length=16000,
#         completion_length=4096,
#     ),
#     BigModelModel(
#         name="glm-4-airx",
#         cost=Cost(prompt=0.00001e-6 * YUAN_TO_USD, completion=0.00001e-6 * YUAN_TO_USD),
#         context_length=8000,
#         completion_length=4096,
#     ),
#     BigModelModel(
#         name="glm-4-flashx",
#         cost=Cost(
#             prompt=0.0000001e-6 * YUAN_TO_USD, completion=0.0000001e-6 * YUAN_TO_USD
#         ),
#         context_length=128000,
#         completion_length=4096,
#     ),
# ]


# class BigModelSDKChatProviderAdapter(OpenAISDKChatAdapter):
#     @staticmethod
#     def get_supported_models() -> list[Model]:
#         return MODELS

#     @staticmethod
#     def get_api_key_name() -> str:
#         return "BIGMODEL_API_KEY"

#     def get_base_sdk_url(self) -> str:
#         return "https://open.bigmodel.cn/api/paas/v4"

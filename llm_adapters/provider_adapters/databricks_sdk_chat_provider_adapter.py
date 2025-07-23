from typing import Any, Optional, Literal, Unpack

from llm_adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from llm_adapters.types import (
    NotGiven,
    ChatCompletionCreateArgs,
    Cost,
    Model,
    Provider,
    Vendor,
)


class DatabricksModel(Model):
    provider_name: str = Provider.databricks.value

    supports_vision: bool = False
    can_assistant_first: bool = False
    supports_stop: bool = False
    can_assistant_last: bool = False
    can_system_last: bool = False
    can_system_multiple: bool = False
    supports_completion: bool = False
    can_system_only: bool = False
    can_system_repeating: bool = False


DBU_USD_RATE = 0.07

MODELS: list[Model] = [
    DatabricksModel(
        name="databricks-meta-llama-3-1-405b-instruct",
        cost=Cost(prompt=71.429 * DBU_USD_RATE, completion=214.286 * DBU_USD_RATE),
        context_length=128000,
        vendor_name=Vendor.meta_llama.value,
    ),
    DatabricksModel(
        name="databricks-llama-4-maverick",
        cost=Cost(prompt=71.423 * DBU_USD_RATE, completion=21.429 * DBU_USD_RATE),
        context_length=128000,
        vendor_name=Vendor.meta_llama.value,
        supports_n=False,
        supports_tools=False,
        supports_tools_choice=False,
    ),
    # TODO: does not except non-integer max token (NULL in our case)
    # DatabricksModel(
    #     name="databricks-claude-3-7-sonnet",
    #     cost=Cost(prompt=42.857 * DBU_USD_RATE, completion=214.286 * DBU_USD_RATE),
    #     context_length=128000,
    #     vendor_name=Vendor.anthropic.value,
    # ),
    DatabricksModel(
        name="databricks-meta-llama-3-3-70b-instruct",
        cost=Cost(prompt=7.143 * DBU_USD_RATE, completion=21.429 * DBU_USD_RATE),
        context_length=128000,
        vendor_name=Vendor.meta_llama.value,
    ),
]


class DatabricksSDKChatProviderAdapter(OpenAISDKChatAdapter):
    @staticmethod
    def get_supported_models() -> list[Model]:
        return MODELS

    @staticmethod
    def get_api_key_name() -> str:
        return "DATABRICKS_API_KEY"

    def get_base_sdk_url(self) -> str:
        return "https://adb-8736858266948228.8.azuredatabricks.net/serving-endpoints"

    def _get_params(
        self,
        *,
        stream: Optional[Literal[False]] | Literal[True] | NotGiven,
        **kwargs: Unpack[ChatCompletionCreateArgs],
    ) -> dict[str, Any]:
        params = super()._get_params(stream=stream, **kwargs)

        messages = params["messages"]
        tools = kwargs.get("tools")
        databricksTools = list(tools) if tools is not None else None

        if databricksTools and not databricksTools[0]["function"].get("parameters"):
            databricksTools[0]["function"]["parameters"] = {
                "type": "object",
            }

        return {
            **params,
            "messages": messages,
            "tools": databricksTools,
            "max_tokens": (
                kwargs.get("max_tokens")
                if kwargs.get("max_tokens")
                else self.get_model().completion_length
            ),
        }

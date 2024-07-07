import re
from typing import Any, Dict, Pattern

from adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from adapters.abstract_adapters.provider_adapter_mixin import ProviderAdapterMixin
from adapters.types import Conversation, ConversationRole, Cost, Model

PROVIDER_NAME = "databricks"
DATABRICKS_BASE_URL = (
    "https://adb-8736858266948228.8.azuredatabricks.net/serving-endpoints"
)
API_KEY_NAME = "DATABRICKS_API_KEY"
API_KEY_PATTERN = re.compile(r".*")


class DatabricksModel(Model):
    supports_first_assistant: bool = False
    supports_multiple_system: bool = False
    provider_name: str = PROVIDER_NAME


# TODO convert DBUs to Dollars to measeure costs


MODELS = [
    DatabricksModel(
        name="databricks-dbrx-instruct",
        cost=Cost(prompt=14.286e-6, completion=42.857e-6),
        context_length=32000,
        vendor_name="databricks",
    ),
    #     DatabricksModel(
    #         name="databricks-meta-llama-3-70b-instruct",
    #         cost=Cost(prompt=0, completion=0),
    #         context_lenght=0,
    #         vendor_name="databricks",
    #     ),
    #     DatabricksModel(
    #         name="databricks-mixtral-8x7b-instruct",
    #         cost="",
    #         context_lenght=0,
    #         vendor_name="databricks",
    #     ),
    #     DatabricksModel(
    #         name="databricks-llama-2-70b-chat",
    #         cost=Cost(prompt=0, completion=0),
    #         context_length=0,
    #         vendor_name="databricks",
    #     ),
    #     DatabricksModel(
    #         name="databricks-gte-large-en",
    #         cost=Cost(prompt=0, completion=0),
    #         context_length=0,
    #         vendor_name="databricks",
    #     ),
    #     DatabricksModel(
    #         name="databricks-bge-large-en",
    #         cost=Cost(prompt=0, completion=0),
    #         context_length=0,
    #         vendor_name="databricks",
    #     ),
    #     DatabricksModel(
    #         name="databricks-mpt-30b-instruct",
    #         cost=Cost(prompt=0, completion=0),
    #         context_length=0,
    #         vendor_name="",
    #     ),
    #     DatabricksModel(
    #         name="databricks-mpt-7b-instruct",
    #         cost=Cost(prompt=0, completion=0),
    #         context_length=0,
    #         vendor_name="databricks",
    #     ),
]


class DatabricksSDKChatProviderAdapter(ProviderAdapterMixin, OpenAISDKChatAdapter):
    # TODO: supports assistant first FALSE
    def get_base_sdk_url(self) -> str:
        return DATABRICKS_BASE_URL

    @staticmethod
    def get_supported_models():
        return MODELS

    @staticmethod
    def get_provider_name() -> str:
        return PROVIDER_NAME

    @staticmethod
    def get_api_key_name() -> str:
        return API_KEY_NAME

    @staticmethod
    def get_api_key_pattern() -> Pattern:
        return API_KEY_PATTERN

    def get_params(
        self,
        llm_input: Conversation,
        **kwargs,
    ) -> Dict[str, Any]:
        params = super().get_params(llm_input, **kwargs)

        messages = params["messages"]

        # Databricks only support system as a first optional message
        if messages and messages[0]["role"] == ConversationRole.system:
            system_message = messages[0]
            messages = [system_message] + [
                msg for msg in messages[1:] if msg["role"] != ConversationRole.system
            ]
        else:
            messages = [
                msg for msg in messages if msg["role"] != ConversationRole.system
            ]

        # Databricks only support ending messages with user or tool roles
        if messages and messages[-1] not in [
            ConversationRole.user,
            ConversationRole.tool,
        ]:
            messages = messages + [{"role": ConversationRole.user, "content": ""}]

        print(kwargs)
        return {
            **params,
            "messages": messages,
            "max_tokens": (
                kwargs.get("max_tokens")
                if kwargs.get("max_tokens")
                else self.get_model().completion_length
            ),
        }

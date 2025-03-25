from enum import Enum
from typing import (
    Optional,
)

from pydantic import BaseModel, Field


class Provider(str, Enum):
    ai21 = "ai21"
    anthropic = "anthropic"
    azure = "azure"
    cerebras = "cerebras"
    cohere = "cohere"
    databricks = "databricks"
    deepinfra = "deepinfra"
    fireworks = "fireworks"
    gemini = "gemini"
    groq = "groq"
    lepton = "lepton"
    moonshot = "moonshot"
    octoai = "octoai"
    openai = "openai"
    openrouter = "openrouter"
    perplexity = "perplexity"
    together = "together"
    vertex = "vertex"
    bigmodel = "bigmodel"
    tensoropera = "tensoropera"
    xai = "xai"
    deepseek = "deepseek"
    lambdalabs = "lambdalabs"


class Vendor(str, Enum):
    nvidia = "nvidia"
    meta_llama = "meta-llama"
    perplexity = "perplexity"
    databricks = "databricks"
    mistralai = "mistralai"
    mixtral = "mixtral"
    pygmalionai = "pygmalionai"
    qwen = "qwen"
    Qwen = "Qwen"  # For Deepseek
    openai = "openai"
    anthropic = "anthropic"
    cohere = "cohere"
    gemini = "gemini"
    together = "together"
    nous_hermes = "nous-hermes"
    hermes_llama = "hermes-llama"
    moonshot = "moonshot"
    wizardlm = "wizardlm"
    google = "google"
    ai21 = "ai21"
    O1 = "01"
    starcoder = "starcoder"
    gryphe = "gryphe"
    microsoft = "microsoft"
    moescape = "moescape"
    bigmodel = "bigmodel"
    deepseek_ai = "deepseek-ai"
    grok = "grok"
    liquid = "liquid"
    deepseek = "deepseek"


class ConversationRole(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"
    function = "function"
    tool = "tool"
    developer = "developer"


class AdapterFinishReason(str, Enum):
    stop = "stop"
    length = "length"
    tool_calls = "tool_calls"
    content_filter = "content_filter"
    function_call = "function_call"


class ContentType(str, Enum):
    text = "text"
    image_url = "image_url"


class Cost(BaseModel):
    prompt: float
    completion: float
    request: float = 0.0


class ModelProperties(BaseModel):
    open_source: bool = False
    chinese: bool = False
    gdpr_compliant: bool = False
    is_nsfw: bool = False


class Model(BaseModel):
    test_async: bool = True

    name: str
    api_name: Optional[str] = None
    vendor_name: str
    provider_name: str
    cost: Cost
    context_length: int
    completion_length: Optional[int] = None

    is_response: bool = False

    supports_chat: bool = True
    supports_completion: bool = True
    supports_functions: bool = True

    supports_streaming: bool = True
    supports_vision: bool = True
    supports_n: bool = True
    supports_tools: bool = True
    supports_tools_streaming: bool = False
    supports_tools_choice: bool = True
    supports_tools_choice_required: bool = True
    supports_json_output: bool = True
    supports_json_content: bool = True
    supports_stop: bool = True
    supports_max_tokens: bool = True
    supports_max_completion_tokens: bool = True

    can_user: bool = True
    can_vision_multiple: bool = True
    can_repeating_roles: bool = True
    can_empty_content: bool = True
    can_temperature: bool = True
    can_presence_penalty: bool = True
    can_repetition_penalty: bool = True
    can_top_p: bool = True
    can_top_k: bool = True
    can_min_p: bool = True

    can_system: bool = True
    can_system_only: bool = True
    can_system_multiple: bool = True
    can_system_last: bool = True

    can_assistant_only: bool = True
    can_assistant_first: bool = True
    can_assistant_last: bool = True

    properties: ModelProperties = Field(default_factory=ModelProperties)

    def get_path(self) -> str:
        return f"{self.provider_name}/{self.vendor_name}/{self.name}"

    def get_api_path(self) -> str:
        return self.api_name if self.api_name else self.name

    def __str__(self) -> str:
        return self.get_path()


__all__ = [
    "Provider",
    "Vendor",
    "ConversationRole",
    "AdapterFinishReason",
    "ContentType",
    "Cost",
    "ModelProperties",
    "Model",
]

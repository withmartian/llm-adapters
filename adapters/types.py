from enum import Enum
from typing import Any, AsyncGenerator, Dict, Generator, List, Literal, Optional, Union

from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion_message import FunctionCall
from pydantic import BaseModel, ConfigDict, Field


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


class Vendor(str, Enum):
    meta_llama = "meta-llama"
    perplexity = "perplexity"
    databricks = "databricks"
    mistralai = "mistralai"
    mixtral = "mixtral"
    pygmalionai = "pygmalionai"
    qwen = "qwen"
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


class ConversationRole(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"
    function = "function"
    tool = "tool"


class AdapterFinishReason(str, Enum):
    stop = "stop"
    length = "length"
    tool_calls = "tool_calls"
    content_filter = "content_filter"
    function_call = "function_call"


class Turn(BaseModel, use_enum_values=True):
    role: Union[ConversationRole]
    content: str


class FunctionOutputTurn(BaseModel, use_enum_values=True):
    role: Literal[ConversationRole.function] = ConversationRole.function
    content: Optional[str] = None
    name: str


class ToolOutputTurn(BaseModel, use_enum_values=True):
    role: Literal[ConversationRole.tool] = ConversationRole.tool
    content: Optional[str] = None
    tool_call_id: str


class FunctionCallTurn(BaseModel, use_enum_values=True):
    role: Literal[ConversationRole.assistant] = ConversationRole.assistant
    function_call: FunctionCall
    content: None = None


class ToolsCallTurn(BaseModel, use_enum_values=True):
    role: Literal[ConversationRole.assistant] = ConversationRole.assistant
    tool_calls: list[ChatCompletionMessageToolCall]
    content: None = None


class ImageDetailsType(str, Enum):
    high = "high"
    low = "low"
    auto = "auto"


class ContentType(str, Enum):
    text = "text"
    image_url = "image_url"


class VisionImageDetails(BaseModel, use_enum_values=True):
    url: str
    details: Optional[ImageDetailsType] = ImageDetailsType.auto


class TextContentEntry(BaseModel, use_enum_values=True):
    type: Literal[ContentType.text] = ContentType.text
    text: str


class ImageContentEntry(BaseModel, use_enum_values=True):
    type: Literal[ContentType.image_url] = ContentType.image_url
    image_url: VisionImageDetails


class ContentTurn(BaseModel, use_enum_values=True, validate_assignment=True):
    role: str = ConversationRole.user
    content: List[Union[ImageContentEntry, TextContentEntry]]


class Cost(BaseModel):
    prompt: float
    completion: float
    request: float = 0.0


class ModelProperties(BaseModel):
    open_source: bool = False
    chinese: bool = False
    gdpr_compliant: bool = False
    is_nsfw: bool = False


# Add test cases for all of them
class Model(BaseModel):
    test_async: bool = True

    name: str
    vendor_name: str
    provider_name: str
    cost: Cost
    context_length: int
    completion_length: Optional[int] = None

    supports_user: bool = True
    supports_repeating_roles: bool = True
    supports_streaming: bool = True
    supports_vision: bool = True
    supports_functions: bool = True
    supports_tools: bool = True
    supports_n: bool = True
    supports_system: bool = True
    supports_multiple_system: bool = True
    supports_empty_content: bool = True
    supports_tool_choice_required: bool = True
    supports_json_output: bool = True
    supports_json_content: bool = True
    supports_last_assistant: bool = True
    supports_first_assistant: bool = True
    supports_temperature: bool = True

    properties: ModelProperties = Field(default_factory=ModelProperties)

    def get_path(self) -> str:
        return f"{self.provider_name}/{self.vendor_name}/{self.name}"

    def _get_api_path(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.get_path()


TurnType = Union[
    Turn,
    FunctionOutputTurn,
    ToolOutputTurn,
    ToolsCallTurn,
    ContentTurn,
    FunctionCallTurn,
]


class Conversation(BaseModel):
    turns: List[TurnType]

    def __init__(
        self,
        turns: Union[
            "Conversation",
            List[TurnType],
            Dict[str, List[TurnType]],
        ],
    ):
        if isinstance(turns, Conversation):
            turns = turns.turns
        elif isinstance(turns, dict) and "turns" in turns:
            turns = turns["turns"]
        super().__init__(turns=turns)

    def __getitem__(self, index: int) -> TurnType:
        return self.turns[index]

    def __setitem__(self, index: int, value: TurnType) -> None:
        if not isinstance(value, Turn):
            raise ValueError("Value must be an instance of Turn")
        self.turns[index] = value

    def __len__(self) -> int:
        return len(self.turns)

    def __iter__(self) -> Any:
        return iter(self.turns)

    def is_last_turn_vision_query(self) -> bool:
        if len(self.turns):
            contentTurn = self.turns[len(self.turns) - 1]
        else:
            return False

        if isinstance(contentTurn, ContentTurn):
            return any(
                content.type == ContentType.image_url for content in contentTurn.content
            )

        return False

    def convert_to_prompt(self) -> "Prompt":
        return Prompt("".join([f"{turn.role}: {turn.content}" for turn in self.turns]))


class AdapterChatCompletion(ChatCompletion):
    cost: float

    # Deprecated. Use choices
    response: Turn
    # Deprecated. Use usage
    token_counts: Optional[Cost] = None


class AdapterChatCompletionChunk(ChatCompletionChunk):
    pass


class AdapterStreamChatCompletion(BaseModel):
    response: Union[
        Generator[AdapterChatCompletionChunk, Any, None],
        AsyncGenerator[AdapterChatCompletionChunk, Any],
    ]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class AdapterStreamSyncChatCompletion(AdapterStreamChatCompletion):
    response: Generator[AdapterChatCompletionChunk, Any, None]


class AdapterStreamAsyncChatCompletion(AdapterStreamChatCompletion):
    response: AsyncGenerator[AdapterChatCompletionChunk, Any]


class AdapterException(Exception):
    pass


class AdapterRateLimitException(AdapterException):
    pass


# Deprecated
class Prompt(str):
    def convert_to_conversation(self) -> Conversation:
        return Conversation(turns=[Turn(role=ConversationRole.user, content=self)])


# Deprecated, Use AdapterChatCompletion
AdapterResponse = AdapterChatCompletion

# Deprecated, Use AdapterStreamChatCompletion
AdapterStreamResponse = AdapterStreamChatCompletion

__all__ = [
    "Conversation",
    "Turn",
    "TurnType",
    "ContentTurn",
    "FunctionOutputTurn",
    "ToolOutputTurn",
    "ToolsCallTurn",
    "FunctionCallTurn",
    "Cost",
    "Model",
    "AdapterChatCompletion",
    "AdapterStreamChatCompletion",
    "AdapterStreamSyncChatCompletion",
    "AdapterStreamAsyncChatCompletion",
    "AdapterChatCompletionChunk",
    "AdapterException",
    "AdapterRateLimitException",
    "Prompt",
    "AdapterResponse",
    "AdapterStreamResponse",
]

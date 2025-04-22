from typing import (
    Any,
    AsyncGenerator,
    Generator,
    Optional,
    Union,
    Dict,
    Iterable,
    List,
    Literal,
)

from llm_adapters.types import (
    Completion,
    ChatCompletionChunk,
    ChatCompletion,
    Cost,
    ChatCompletionMessageToolCall,
    ChatCompletionMessageParam,
    ContentType,
)

from pydantic import BaseModel, ConfigDict
from enum import Enum
from openai.types.chat.chat_completion_message import FunctionCall

from llm_adapters.types.model import ConversationRole

from pydantic_core import core_schema


class Turn(BaseModel, use_enum_values=True):
    role: Union[ConversationRole, str]
    content: str


class AdapterChatCompletion(ChatCompletion):
    cost: float

    # V7
    response: Turn
    # V7
    token_counts: Cost


class AdapterChatCompletionChunk(ChatCompletionChunk):
    pass


# V8
# AdapterStreamSyncChatCompletion = Generator[AdapterChatCompletionChunk, Any, None]
# V7
class AdapterStreamSyncChatCompletion:
    def __init__(
        self, response: Generator[AdapterChatCompletionChunk, Any, None]
    ) -> None:
        self.response = response

    def __iter__(self) -> "AdapterStreamSyncChatCompletion":
        return self

    def __next__(self) -> AdapterChatCompletionChunk:
        return next(self.response)

    def close(self) -> None:
        self.response.close()

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: Any
    ) -> core_schema.CoreSchema:
        # Explicitly tell Pydantic to treat this as an arbitrary type
        return core_schema.no_info_after_validator_function(
            cls._validate, core_schema.any_schema()
        )

    @staticmethod
    def _validate(value: Any) -> "AdapterStreamSyncChatCompletion":
        assert isinstance(
            value, AdapterStreamSyncChatCompletion
        ), f"Expected AdapterStreamSyncChatCompletion, got {type(value)}"
        return value


# V8
# AdapterStreamAsyncChatCompletion = AsyncGenerator[AdapterChatCompletionChunk, Any]
# V7
class AdapterStreamAsyncChatCompletion:
    def __init__(
        self, response: AsyncGenerator[AdapterChatCompletionChunk, Any]
    ) -> None:
        self.response = response

    def __aiter__(self) -> "AdapterStreamAsyncChatCompletion":
        return self

    async def __anext__(self) -> AdapterChatCompletionChunk:
        return await self.response.__anext__()

    async def aclose(self) -> None:
        await self.response.aclose()

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: Any
    ) -> core_schema.CoreSchema:
        # Explicitly tell Pydantic to treat this as an arbitrary type
        return core_schema.no_info_after_validator_function(
            cls._validate, core_schema.any_schema()
        )

    @staticmethod
    def _validate(value: Any) -> "AdapterStreamAsyncChatCompletion":
        assert isinstance(
            value, AdapterStreamAsyncChatCompletion
        ), f"Expected AdapterStreamAsyncChatCompletion, got {type(value)}"
        return value


# V8
# AdapterStreamChatCompletion = Union[
#     AdapterStreamSyncChatCompletion,
#     AdapterStreamAsyncChatCompletion,
# ]
# V7
class AdapterStreamChatCompletion(BaseModel):
    response: Union[
        AdapterStreamSyncChatCompletion,
        AdapterStreamAsyncChatCompletion,
    ]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class AdapterCompletion(Completion):
    cost: float


class AdapterCompletionChunk(Completion):
    pass


class AdapterStreamCompletion(BaseModel):
    response: Union[
        Generator[AdapterCompletionChunk, Any, None],
        AsyncGenerator[AdapterCompletionChunk, Any],
    ]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class AdapterStreamSyncCompletion(AdapterStreamCompletion):
    response: Generator[AdapterCompletionChunk, Any, None]


class AdapterStreamAsyncCompletion(AdapterStreamCompletion):
    response: AsyncGenerator[AdapterCompletionChunk, Any]


# Deprecated


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
            Dict[str, List[Union[TurnType, Dict[str, Any]]]],
        ],
    ):
        if isinstance(turns, Conversation):
            turns = turns.turns
        elif isinstance(turns, dict) and "turns" in turns:
            turns = [
                Turn(**turn) if isinstance(turn, dict) else turn
                for turn in turns["turns"]
            ]
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

    def convert_to_openai_format(self) -> Iterable[ChatCompletionMessageParam]:
        return self.model_dump()["turns"]  # type: ignore


# Deprecated
class Prompt(str):
    def convert_to_conversation(self) -> Conversation:
        return Conversation(turns=[Turn(role=ConversationRole.user, content=self)])


# Deprecated, Use AdapterChatCompletion
AdapterResponse = AdapterChatCompletion

# Deprecated, Use AdapterStreamChatCompletion
AdapterStreamResponse = AdapterStreamChatCompletion


__all__ = [
    "AdapterChatCompletion",
    "AdapterStreamChatCompletion",
    "AdapterStreamSyncChatCompletion",
    "AdapterStreamAsyncChatCompletion",
    "AdapterChatCompletionChunk",
    "AdapterCompletion",
    "AdapterCompletionChunk",
    "AdapterStreamCompletion",
    "AdapterStreamSyncCompletion",
    "AdapterStreamAsyncCompletion",
    # Deprecated
    "Turn",
    "FunctionOutputTurn",
    "ToolOutputTurn",
    "ToolsCallTurn",
    "ImageDetailsType",
    "VisionImageDetails",
    "TextContentEntry",
    "ImageContentEntry",
    "ContentTurn",
    "TurnType",
    "Conversation",
    "Prompt",
    "AdapterResponse",
    "AdapterStreamResponse",
]

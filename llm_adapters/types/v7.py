# from enum import Enum
# from typing import (
#     Any,
#     Dict,
#     Iterable,
#     List,
#     Literal,
#     Optional,
#     Union,
# )
# from pydantic import BaseModel
# from llm_adapters.types import (
#     AdapterStreamChatCompletion,
#     AdapterChatCompletion,
#     ConversationRole,
#     ChatCompletionMessageToolCall,
#     ChatCompletionMessageParam,
# )
# from openai.types.chat.chat_completion_message import FunctionCall


# class Turn(BaseModel, use_enum_values=True):
#     role: Union[ConversationRole]
#     content: str


# class FunctionOutputTurn(BaseModel, use_enum_values=True):
#     role: Literal[ConversationRole.function] = ConversationRole.function
#     content: Optional[str] = None
#     name: str


# class ToolOutputTurn(BaseModel, use_enum_values=True):
#     role: Literal[ConversationRole.tool] = ConversationRole.tool
#     content: Optional[str] = None
#     tool_call_id: str


# class FunctionCallTurn(BaseModel, use_enum_values=True):
#     role: Literal[ConversationRole.assistant] = ConversationRole.assistant
#     function_call: FunctionCall
#     content: None = None


# class ToolsCallTurn(BaseModel, use_enum_values=True):
#     role: Literal[ConversationRole.assistant] = ConversationRole.assistant
#     tool_calls: list[ChatCompletionMessageToolCall]
#     content: None = None


# class ImageDetailsType(str, Enum):
#     high = "high"
#     low = "low"
#     auto = "auto"


# class ContentType(str, Enum):
#     text = "text"
#     image_url = "image_url"


# class VisionImageDetails(BaseModel, use_enum_values=True):
#     url: str
#     details: Optional[ImageDetailsType] = ImageDetailsType.auto


# class TextContentEntry(BaseModel, use_enum_values=True):
# #     type: Literal[ContentType.text] = ContentType.text
#     text: str


# class ImageContentEntry(BaseModel, use_enum_values=True):
# #     type: Literal[ContentType.image_url] = ContentType.image_url
#     image_url: VisionImageDetails


# class ContentTurn(BaseModel, use_enum_values=True, validate_assignment=True):
#     role: str = ConversationRole.user
#     content: List[Union[ImageContentEntry, TextContentEntry]]


# TurnType = Union[
#     Turn,
#     FunctionOutputTurn,
#     ToolOutputTurn,
#     ToolsCallTurn,
#     ContentTurn,
#     FunctionCallTurn,
# ]


# class Conversation(BaseModel):
#     turns: List[TurnType]

#     def __init__(
#         self,
#         turns: Union[
#             "Conversation",
#             List[TurnType],
#             Dict[str, List[TurnType]],
#         ],
#     ):
#         if isinstance(turns, Conversation):
#             turns = turns.turns
#         elif isinstance(turns, dict) and "turns" in turns:
#             turns = turns["turns"]
#         super().__init__(turns=turns)

#     def __getitem__(self, index: int) -> TurnType:
#         return self.turns[index]

#     def __setitem__(self, index: int, value: TurnType) -> None:
#         if not isinstance(value, Turn):
#             raise ValueError("Value must be an instance of Turn")
#         self.turns[index] = value

#     def __len__(self) -> int:
#         return len(self.turns)

#     def __iter__(self) -> Any:
#         return iter(self.turns)

#     def is_last_turn_vision_query(self) -> bool:
#         if len(self.turns):
#             contentTurn = self.turns[len(self.turns) - 1]
#         else:
#             return False

#         if isinstance(contentTurn, ContentTurn):
#             return any(
#                 content.type == ContentType.image_url for content in contentTurn.content
#             )

#         return False

#     def convert_to_prompt(self) -> "Prompt":
#         return Prompt("".join([f"{turn.role}: {turn.content}" for turn in self.turns]))

#     def convert_to_openai_format(self) -> Iterable[ChatCompletionMessageParam]:
#         return self.model_dump()["turns"]  # type: ignore


# # Deprecated
# class Prompt(str):
#     def convert_to_conversation(self) -> Conversation:
#         return Conversation(turns=[Turn(role=ConversationRole.user, content=self)])


# # Deprecated, Use AdapterChatCompletion
# AdapterResponse = AdapterChatCompletion

# # Deprecated, Use AdapterStreamChatCompletion
# AdapterStreamResponse = AdapterStreamChatCompletion

# __all__ = [
#     "Turn",
#     "FunctionOutputTurn",
#     "ToolOutputTurn",
#     "ToolsCallTurn",
#     "ImageDetailsType",
#     "ContentType",
#     "VisionImageDetails",
#     "TextContentEntry",
#     "ImageContentEntry",
#     "ContentTurn",
#     "TurnType",
#     "Conversation",
#     "Prompt",
#     "AdapterResponse",
#     "AdapterStreamResponse",
# ]

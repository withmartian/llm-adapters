from typing import (
    Any,
    AsyncGenerator,
    Generator,
    Union,
)

from llm_adapters.types import Completion, ChatCompletionChunk, ChatCompletion
from pydantic import BaseModel, ConfigDict


class AdapterChatCompletion(ChatCompletion):
    cost: float


class AdapterChatCompletionChunk(ChatCompletionChunk):
    pass


AdapterStreamSyncChatCompletion = Generator[AdapterChatCompletionChunk, Any, None]

AdapterStreamAsyncChatCompletion = AsyncGenerator[AdapterChatCompletionChunk, Any]

AdapterStreamChatCompletion = Union[
    AdapterStreamSyncChatCompletion,
    AdapterStreamAsyncChatCompletion,
]


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
]

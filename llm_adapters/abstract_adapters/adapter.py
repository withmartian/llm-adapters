from abc import ABC, abstractmethod
from typing import Literal, Optional, Unpack, overload

from openai import NOT_GIVEN, NotGiven

from llm_adapters.types import (
    AdapterChatCompletion,
    AdapterCompletion,
    AdapterStreamAsyncChatCompletion,
    AdapterStreamAsyncCompletion,
    AdapterStreamSyncChatCompletion,
    AdapterStreamSyncCompletion,
    Model,
)
from llm_adapters.types.request import (
    ChatCompletionCreateArgs,
    CompletionCreateArgs,
)


class Adapter(ABC):
    def __str__(self) -> str:
        return f"adapter-{self.get_model().get_path()}"

    def set_api_key(self, api_key: str) -> None:
        pass

    @abstractmethod
    def get_model(self) -> Model:
        pass

    @overload
    def execute_completion_sync(
        self,
        *,
        stream: Literal[True],
        **kwargs: Unpack[CompletionCreateArgs],
    ) -> AdapterStreamSyncCompletion: ...
    @overload
    def execute_completion_sync(
        self,
        *,
        stream: Optional[Literal[False]] | NotGiven = NOT_GIVEN,
        **kwargs: Unpack[CompletionCreateArgs],
    ) -> AdapterCompletion: ...
    @abstractmethod
    def execute_completion_sync(
        self,
        *,
        stream: Optional[Literal[False]] | Literal[True] | NotGiven = NOT_GIVEN,
        **kwargs: Unpack[CompletionCreateArgs],
    ) -> AdapterCompletion | AdapterStreamSyncCompletion: ...

    @overload
    async def execute_completion_async(
        self,
        *,
        stream: Literal[True],
        **kwargs: Unpack[CompletionCreateArgs],
    ) -> AdapterStreamAsyncCompletion: ...
    @overload
    async def execute_completion_async(
        self,
        *,
        stream: Optional[Literal[False]] | NotGiven = NOT_GIVEN,
        **kwargs: Unpack[CompletionCreateArgs],
    ) -> AdapterCompletion: ...
    @abstractmethod
    async def execute_completion_async(
        self,
        *,
        stream: Optional[Literal[False]] | Literal[True] | NotGiven = NOT_GIVEN,
        **kwargs: Unpack[CompletionCreateArgs],
    ) -> AdapterCompletion | AdapterStreamAsyncCompletion: ...

    @overload
    def execute_chat_completion_sync(
        self,
        *,
        stream: Literal[True],
        **kwargs: Unpack[ChatCompletionCreateArgs],
    ) -> AdapterStreamSyncChatCompletion: ...
    @overload
    def execute_chat_completion_sync(
        self,
        *,
        stream: Optional[Literal[False]] | NotGiven = NOT_GIVEN,
        **kwargs: Unpack[ChatCompletionCreateArgs],
    ) -> AdapterChatCompletion: ...
    @abstractmethod
    def execute_chat_completion_sync(
        self,
        *,
        stream: Optional[Literal[False]] | Literal[True] | NotGiven = NOT_GIVEN,
        **kwargs: Unpack[ChatCompletionCreateArgs],
    ) -> AdapterChatCompletion | AdapterStreamSyncChatCompletion: ...

    @overload
    async def execute_chat_completion_async(
        self,
        *,
        stream: Literal[True],
        **kwargs: Unpack[ChatCompletionCreateArgs],
    ) -> AdapterStreamAsyncChatCompletion: ...
    @overload
    async def execute_chat_completion_async(
        self,
        *,
        stream: Optional[Literal[False]] | NotGiven = NOT_GIVEN,
        **kwargs: Unpack[ChatCompletionCreateArgs],
    ) -> AdapterChatCompletion: ...
    @abstractmethod
    async def execute_chat_completion_async(
        self,
        *,
        stream: Optional[Literal[False]] | Literal[True] | NotGiven = NOT_GIVEN,
        **kwargs: Unpack[ChatCompletionCreateArgs],
    ) -> AdapterChatCompletion | AdapterStreamAsyncChatCompletion: ...

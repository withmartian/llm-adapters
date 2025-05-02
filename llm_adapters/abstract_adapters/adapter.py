from abc import ABC, abstractmethod
from typing import Any, Iterable, Literal, Optional, Unpack, overload

from openai import NOT_GIVEN, NotGiven

from llm_adapters.types import (
    AdapterChatCompletion,
    AdapterCompletion,
    AdapterStreamAsyncChatCompletion,
    AdapterStreamAsyncCompletion,
    AdapterStreamSyncChatCompletion,
    AdapterStreamSyncCompletion,
    Model,
    ChatCompletionMessageParam,
    Conversation,
    Prompt,
)
from llm_adapters.types.request import (
    ChatCompletionCreateArgs,
    CompletionCreateArgs,
)

# TODO: add Deepseek and pricing

# DeepSeek-V3 at 50% off
# DeepSeek-R1 at a massive 75% off


class Adapter(ABC):
    def __str__(self) -> str:
        return f"adapter-{self.get_model().get_path()}"

    def set_api_key(self, api_key: str) -> None:
        pass

    @abstractmethod
    def get_model(self) -> Model:
        pass

    # Deprecated
    @staticmethod
    def convert_to_input(llm_input: Conversation | Prompt) -> Conversation:
        if isinstance(llm_input, Conversation):
            return llm_input
        return llm_input.convert_to_conversation()

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

    # V7
    @overload
    def execute_sync(
        self,
        messages: Iterable[ChatCompletionMessageParam] | Conversation,
        stream: Literal[True],
        **kwargs: Any,
    ) -> AdapterStreamSyncChatCompletion: ...
    @overload
    def execute_sync(
        self,
        messages: Iterable[ChatCompletionMessageParam] | Conversation,
        stream: Optional[Literal[False]] | NotGiven = NOT_GIVEN,
        **kwargs: Any,
    ) -> AdapterChatCompletion: ...
    def execute_sync(
        self,
        messages: Iterable[ChatCompletionMessageParam] | Conversation,
        stream: Optional[Literal[False]] | Literal[True] | NotGiven = NOT_GIVEN,
        **kwargs: Any,
    ) -> AdapterChatCompletion | AdapterStreamSyncChatCompletion:
        if isinstance(messages, Conversation):
            openai_messages = messages.convert_to_openai_format()
        else:
            openai_messages = messages

        return self.execute_chat_completion_sync(
            messages=openai_messages, stream=stream, **kwargs
        )

    # V7
    @overload
    async def execute_async(
        self,
        messages: Iterable[ChatCompletionMessageParam] | Conversation,
        stream: Literal[True],
        **kwargs: Any,
    ) -> AdapterStreamAsyncChatCompletion: ...
    @overload
    async def execute_async(
        self,
        messages: Iterable[ChatCompletionMessageParam] | Conversation,
        stream: Optional[Literal[False]] | NotGiven = NOT_GIVEN,
        **kwargs: Any,
    ) -> AdapterChatCompletion: ...
    async def execute_async(
        self,
        messages: Iterable[ChatCompletionMessageParam] | Conversation,
        stream: Optional[Literal[False]] | Literal[True] | NotGiven = NOT_GIVEN,
        **kwargs: Any,
    ) -> AdapterChatCompletion | AdapterStreamAsyncChatCompletion:
        if isinstance(messages, Conversation):
            openai_messages = messages.convert_to_openai_format()
        else:
            openai_messages = messages

        return await self.execute_chat_completion_async(
            messages=openai_messages, stream=stream, **kwargs
        )


BaseAdapter = Adapter

__all__ = ["BaseAdapter", "Adapter"]

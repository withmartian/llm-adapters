from typing import Any, Iterable, Literal, Optional, Union, Unpack, cast, overload

from llm_adapters.adapter_factory import AdapterFactory
from llm_adapters.types import (
    ChatCompletionCreateKwargs,
    CompletionCreateKwargs,
    NotGiven,
    NOT_GIVEN,
    AdapterChatCompletion,
    AdapterStreamSyncChatCompletion,
    ChatModel,
    ChatCompletionMessageParam,
    AdapterCompletion,
    AdapterStreamSyncCompletion,
    ChatCompletionCreateArgs,
    CompletionCreateArgs,
)


class Completion:
    @overload
    def create(
        self,
        *,
        model: Union[str, ChatModel],
        prompt: Union[str, list[str], Iterable[int], Iterable[Iterable[int]], None],
        stream: Optional[Literal[False]] | NotGiven = NOT_GIVEN,
        **kwargs: Unpack[CompletionCreateKwargs],
    ) -> AdapterCompletion: ...

    @overload
    def create(
        self,
        *,
        model: Union[str, ChatModel],
        prompt: Union[str, list[str], Iterable[int], Iterable[Iterable[int]], None],
        stream: Literal[True],
        **kwargs: Unpack[CompletionCreateKwargs],
    ) -> AdapterStreamSyncCompletion: ...

    def create(
        self,
        *,
        model: Union[str, ChatModel],
        prompt: Union[str, list[str], Iterable[int], Iterable[Iterable[int]], None],
        stream: Optional[Literal[False]] | Literal[True] | NotGiven = NOT_GIVEN,
        **kwargs: Unpack[CompletionCreateKwargs],
    ) -> AdapterCompletion | AdapterStreamSyncCompletion:
        return AdapterFactory.get_adapter_by_path(model).execute_completion_sync(
            stream=stream,
            **cast(CompletionCreateArgs, {"model": model, "prompt": prompt, **kwargs}),
        )


class ChatCompletion:
    @overload
    def create(
        self,
        *,
        model: Union[str, ChatModel],
        messages: Iterable[ChatCompletionMessageParam],
        stream: Optional[Literal[False]] | NotGiven = NOT_GIVEN,
        **kwargs: Unpack[ChatCompletionCreateKwargs],
    ) -> AdapterChatCompletion: ...

    @overload
    def create(
        self,
        *,
        model: Union[str, ChatModel],
        messages: Iterable[ChatCompletionMessageParam],
        stream: Literal[True],
        **kwargs: Unpack[ChatCompletionCreateKwargs],
    ) -> AdapterStreamSyncChatCompletion: ...

    def create(
        self,
        *,
        model: Union[str, ChatModel],
        messages: Iterable[ChatCompletionMessageParam],
        stream: Optional[Literal[False]] | Literal[True] | NotGiven = NOT_GIVEN,
        **kwargs: Unpack[ChatCompletionCreateKwargs],
    ) -> AdapterChatCompletion | AdapterStreamSyncChatCompletion:
        return AdapterFactory.get_adapter_by_path(model).execute_chat_completion_sync(
            stream=stream,
            **cast(
                ChatCompletionCreateArgs,
                {"model": model, "messages": messages, **kwargs},
            ),
        )


class Chat:
    def __init__(self) -> None:
        self._chat_completion = ChatCompletion()

    @property
    def completion(self) -> ChatCompletion:
        return self._chat_completion


class OpenAI:
    def __init__(self, **kwargs: Any) -> None:
        self._chat = Chat()
        self._completion = Completion()

    @property
    def chat(self) -> Chat:
        return self._chat

    @property
    def completion(self) -> Completion:
        return self._completion


__all__ = ["OpenAI"]

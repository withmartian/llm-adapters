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
    AdapterStreamAsyncChatCompletion,
    AdapterStreamAsyncCompletion,
    FileObject,
    FilePurpose,
)
from llm_adapters.types.files import CreateFileKwargs


class Completions:
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


class AsyncCompletions:
    @overload
    async def create(
        self,
        *,
        model: Union[str, ChatModel],
        prompt: Union[str, list[str], Iterable[int], Iterable[Iterable[int]], None],
        stream: Optional[Literal[False]] | NotGiven = NOT_GIVEN,
        **kwargs: Unpack[CompletionCreateKwargs],
    ) -> AdapterCompletion: ...

    @overload
    async def create(
        self,
        *,
        model: Union[str, ChatModel],
        prompt: Union[str, list[str], Iterable[int], Iterable[Iterable[int]], None],
        stream: Literal[True],
        **kwargs: Unpack[CompletionCreateKwargs],
    ) -> AdapterStreamAsyncCompletion: ...

    async def create(
        self,
        *,
        model: Union[str, ChatModel],
        prompt: Union[str, list[str], Iterable[int], Iterable[Iterable[int]], None],
        stream: Optional[Literal[False]] | Literal[True] | NotGiven = NOT_GIVEN,
        **kwargs: Unpack[CompletionCreateKwargs],
    ) -> AdapterCompletion | AdapterStreamAsyncCompletion:
        return await AdapterFactory.get_adapter_by_path(model).execute_completion_async(
            stream=stream,
            **cast(CompletionCreateArgs, {"model": model, "prompt": prompt, **kwargs}),
        )


class ChatCompletions:
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


class AsyncChatCompletions:
    @overload
    async def create(
        self,
        *,
        model: Union[str, ChatModel],
        messages: Iterable[ChatCompletionMessageParam],
        stream: Optional[Literal[False]] | NotGiven = NOT_GIVEN,
        **kwargs: Unpack[ChatCompletionCreateKwargs],
    ) -> AdapterChatCompletion: ...

    @overload
    async def create(
        self,
        *,
        model: Union[str, ChatModel],
        messages: Iterable[ChatCompletionMessageParam],
        stream: Literal[True],
        **kwargs: Unpack[ChatCompletionCreateKwargs],
    ) -> AdapterStreamAsyncChatCompletion: ...

    async def create(
        self,
        *,
        model: Union[str, ChatModel],
        messages: Iterable[ChatCompletionMessageParam],
        stream: Optional[Literal[False]] | Literal[True] | NotGiven = NOT_GIVEN,
        **kwargs: Unpack[ChatCompletionCreateKwargs],
    ) -> AdapterChatCompletion | AdapterStreamAsyncChatCompletion:
        return await AdapterFactory.get_adapter_by_path(
            model
        ).execute_chat_completion_async(
            stream=stream,
            **cast(
                ChatCompletionCreateArgs,
                {"model": model, "messages": messages, **kwargs},
            ),
        )


class Files:
    def create(
        self,
        *,
        file: FileObject,
        purpose: FilePurpose,
        **kwargs: Unpack[CreateFileKwargs],
    ) -> FileObject:
        return OpenAI().files.create(file=file, purpose=purpose, **kwargs)


class AsyncFiles:
    async def create(
        self,
        *,
        file: FileObject,
        purpose: FilePurpose,
        **kwargs: Unpack[CreateFileKwargs],
    ) -> FileObject:
        return await OpenAI().files.create(file=file, purpose=purpose, **kwargs)


class Chat:
    def __init__(self) -> None:
        self._chat_completions = ChatCompletions()

    @property
    def completions(self) -> ChatCompletions:
        return self._chat_completions


class AsyncChat:
    def __init__(self) -> None:
        self._async_chat_completions = AsyncChatCompletions()

    @property
    def completions(self) -> AsyncChatCompletions:
        return self._async_chat_completions


class OpenAI:
    def __init__(self, **kwargs: Any) -> None:
        self._chat = Chat()
        self._completions = Completions()
        self._files = Files()

    @property
    def chat(self) -> Chat:
        return self._chat

    @property
    def completions(self) -> Completions:
        return self._completions

    @property
    def files(self) -> Files:
        return self._files


class AsyncOpenAI:
    def __init__(self, **kwargs: Any) -> None:
        self._async_chat = AsyncChat()
        self._async_completions = AsyncCompletions()
        self._async_files = AsyncFiles()

    @property
    def chat(self) -> AsyncChat:
        return self._async_chat

    @property
    def completions(self) -> AsyncCompletions:
        return self._async_completions

    @property
    def files(self) -> AsyncFiles:
        return self._async_files


__all__ = ["OpenAI", "AsyncOpenAI"]

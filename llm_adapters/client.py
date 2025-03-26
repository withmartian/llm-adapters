from typing import Any, Iterable, Literal, Optional, Union, Unpack, cast, overload
from openai import OpenAI as OOpenAI
from openai import AsyncOpenAI as AsyncOOpenAI
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
    Metadata,
    Batch,
    HttpxBinaryResponseContent,
    SyncCursorPage,
    AsyncCursorPage,
    AsyncPaginator,
)
from llm_adapters.types.files import ExtraKwargs, FileTypes


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


class Batches:
    def create(
        self,
        *,
        completion_window: Literal["24h"],
        endpoint: Literal["/v1/chat/completions", "/v1/embeddings", "/v1/completions"],
        input_file_id: str,
        metadata: Optional[Metadata] | NotGiven = NOT_GIVEN,
        **kwargs: Unpack[ExtraKwargs],
    ) -> Batch:
        return OOpenAI().batches.create(
            completion_window=completion_window,
            endpoint=endpoint,
            input_file_id=input_file_id,
            metadata=metadata,
            **kwargs,
        )

    def retrieve(self, batch_id: str, **kwargs: Unpack[ExtraKwargs]) -> Batch:
        return OOpenAI().batches.retrieve(batch_id, **kwargs)

    def cancel(self, batch_id: str, **kwargs: Unpack[ExtraKwargs]) -> Batch:
        return OOpenAI().batches.cancel(batch_id, **kwargs)

    def list(
        self,
        *,
        after: str | NotGiven = NOT_GIVEN,
        limit: int | NotGiven = NOT_GIVEN,
        **kwargs: Unpack[ExtraKwargs],
    ) -> SyncCursorPage[Batch]:
        return OOpenAI().batches.list(after=after, limit=limit, **kwargs)


class AsyncBatches:
    async def create(
        self,
        *,
        completion_window: Literal["24h"],
        endpoint: Literal["/v1/chat/completions", "/v1/embeddings", "/v1/completions"],
        input_file_id: str,
        metadata: Optional[Metadata] | NotGiven = NOT_GIVEN,
        **kwargs: Unpack[ExtraKwargs],
    ) -> Batch:
        return await AsyncOOpenAI().batches.create(
            completion_window=completion_window,
            endpoint=endpoint,
            input_file_id=input_file_id,
            metadata=metadata,
            **kwargs,
        )

    async def retrieve(self, batch_id: str, **kwargs: Unpack[ExtraKwargs]) -> Batch:
        return await AsyncOOpenAI().batches.retrieve(batch_id, **kwargs)

    async def cancel(self, batch_id: str, **kwargs: Unpack[ExtraKwargs]) -> Batch:
        return await AsyncOOpenAI().batches.cancel(batch_id, **kwargs)

    async def list(
        self,
        *,
        after: str | NotGiven = NOT_GIVEN,
        limit: int | NotGiven = NOT_GIVEN,
        **kwargs: Unpack[ExtraKwargs],
    ) -> AsyncPaginator[Batch, AsyncCursorPage[Batch]]:
        return AsyncOOpenAI().batches.list(after=after, limit=limit, **kwargs)


class Files:
    def create(
        self,
        *,
        file: FileTypes,
        purpose: FilePurpose,
        **kwargs: Unpack[ExtraKwargs],
    ) -> FileObject:
        return OOpenAI().files.create(file=file, purpose=purpose, **kwargs)

    def content(
        self, file_id: str, **kwargs: Unpack[ExtraKwargs]
    ) -> HttpxBinaryResponseContent:
        return OOpenAI().files.content(file_id, **kwargs)


class AsyncFiles:
    async def create(
        self,
        *,
        file: FileTypes,
        purpose: FilePurpose,
        **kwargs: Unpack[ExtraKwargs],
    ) -> FileObject:
        return await AsyncOOpenAI().files.create(file=file, purpose=purpose, **kwargs)

    async def content(
        self, file_id: str, **kwargs: Unpack[ExtraKwargs]
    ) -> HttpxBinaryResponseContent:
        return await AsyncOOpenAI().files.content(file_id, **kwargs)


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
        self._batches = Batches()

    @property
    def chat(self) -> Chat:
        return self._chat

    @property
    def completions(self) -> Completions:
        return self._completions

    @property
    def files(self) -> Files:
        return self._files

    @property
    def batches(self) -> Batches:
        return self._batches


class AsyncOpenAI:
    def __init__(self, **kwargs: Any) -> None:
        self._async_chat = AsyncChat()
        self._async_completions = AsyncCompletions()
        self._async_files = AsyncFiles()
        self._async_batches = AsyncBatches()

    @property
    def chat(self) -> AsyncChat:
        return self._async_chat

    @property
    def completions(self) -> AsyncCompletions:
        return self._async_completions

    @property
    def files(self) -> AsyncFiles:
        return self._async_files

    @property
    def batches(self) -> AsyncBatches:
        return self._async_batches


__all__ = ["OpenAI", "AsyncOpenAI"]

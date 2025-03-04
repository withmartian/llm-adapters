from typing import (
    Iterable,
    Literal,
    Optional,
    TypedDict,
    Union,
)

import httpx

from llm_adapters.types import (
    NotGiven,
    ChatCompletionMessageParam,
    ChatModel,
    ChatCompletionAudioParam,
    FunctionCall,
    FunctionCreate,
    ChatCompletionModality,
    ChatCompletionPredictionContentParam,
    ChatCompletionReasoningEffort,
    ResponseFormat,
    ChatCompletionStreamOptionsParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
)

from llm_adapters.types.openai import Body, Headers, Query


class CompletionCreateKwargs(TypedDict, total=False):
    # model: Union[str]
    # prompt: Union[str, list[str], Iterable[int], Iterable[Iterable[int]], None]
    best_of: Optional[int] | NotGiven
    echo: Optional[bool] | NotGiven
    frequency_penalty: Optional[float] | NotGiven
    logit_bias: Optional[dict[str, int]] | NotGiven
    logprobs: Optional[int] | NotGiven
    max_tokens: Optional[int] | NotGiven
    n: Optional[int] | NotGiven
    presence_penalty: Optional[float] | NotGiven
    seed: Optional[int] | NotGiven
    stop: Union[Optional[str], list[str], None] | NotGiven
    # stream: Optional[Literal[False]] | NotGiven
    stream_options: Optional[ChatCompletionStreamOptionsParam] | NotGiven
    suffix: Optional[str] | NotGiven
    temperature: Optional[float] | NotGiven
    top_p: Optional[float] | NotGiven
    user: str | NotGiven

    extra_headers: Headers | None
    extra_query: Query | None
    extra_body: Body | None
    timeout: float | httpx.Timeout | None | NotGiven


class CompletionCreateArgs(CompletionCreateKwargs, total=False):
    model: Union[str]
    prompt: Union[str, list[str], Iterable[int], Iterable[Iterable[int]], None]
    # stream: Optional[Literal[False]] | NotGiven


class ChatCompletionCreateKwargs(TypedDict, total=False):
    # messages: Iterable[ChatCompletionMessageParam]
    # model: Union[str, ChatModel]
    audio: Optional[ChatCompletionAudioParam] | None
    frequency_penalty: Optional[float] | None
    function_call: FunctionCall | None
    functions: Iterable[FunctionCreate] | None
    logit_bias: Optional[dict[str, int]] | None
    logprobs: Optional[bool] | None
    max_completion_tokens: Optional[int] | None
    max_tokens: Optional[int] | None
    metadata: Optional[dict[str, str]] | None
    modalities: Optional[list[ChatCompletionModality]] | None
    n: Optional[int] | None
    parallel_tool_calls: bool | None
    prediction: Optional[ChatCompletionPredictionContentParam] | None
    presence_penalty: Optional[float] | None
    reasoning_effort: ChatCompletionReasoningEffort | None
    response_format: ResponseFormat | None
    seed: Optional[int] | None
    service_tier: Optional[Literal["auto", "default"]] | None
    stop: Union[Optional[str], list[str]] | None
    store: Optional[bool] | None
    # stream: Optional[Literal[False]] | None
    stream_options: Optional[ChatCompletionStreamOptionsParam] | None
    temperature: Optional[float] | None
    tool_choice: ChatCompletionToolChoiceOptionParam | None
    tools: Iterable[ChatCompletionToolParam] | None
    top_logprobs: Optional[int] | None
    top_p: Optional[float] | None
    user: str | None

    # min_p: Optional[float] | None

    extra_headers: Headers | None
    extra_query: Query | None
    extra_body: Body | None
    timeout: float | httpx.Timeout | None


class ChatCompletionCreateArgs(ChatCompletionCreateKwargs):
    messages: Iterable[ChatCompletionMessageParam]
    model: Union[str, ChatModel]
    # stream: Optional[Literal[False]] | NotGiven


__all__ = [
    "ChatCompletionCreateKwargs",
    "CompletionCreateKwargs",
    "ChatCompletionCreateArgs",
    "CompletionCreateArgs",
]

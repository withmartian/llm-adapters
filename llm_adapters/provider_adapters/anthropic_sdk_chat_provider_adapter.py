from enum import Enum
import json
import time
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    TypedDict,
    Union,
    Unpack,
    cast,
)

from anthropic import Anthropic, AsyncAnthropic
from anthropic.types import (
    Message,
    RawContentBlockDeltaEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStreamEvent,
)
from anthropic.types.message_create_params import (
    Metadata,
    ToolChoice,
    ToolChoiceToolChoiceAny,
    ToolChoiceToolChoiceAuto,
    ToolChoiceToolChoiceTool,
)
from anthropic.types.message_param import MessageParam
from anthropic.types.text_delta import TextDelta
from anthropic.types.text_block_param import TextBlockParam
from anthropic.types.tool_param import ToolParam
from httpx import AsyncClient, Client, Limits, Timeout

from openai.types.chat.chat_completion_chunk import Choice as ChoiceChunk, ChoiceDelta

from llm_adapters.abstract_adapters.sdk_chat_adapter import SDKChatAdapter
from llm_adapters.constants import (
    ADAPTERS_ENABLE_CACHE_PRICING,
    HTTP_CONNECT_TIMEOUT,
    HTTP_TIMEOUT,
    MAX_CONNECTIONS_PER_PROCESS,
    MAX_KEEPALIVE_CONNECTIONS_PER_PROCESS,
)
from llm_adapters.general_utils import delete_none_values, process_image_url_anthropic
from llm_adapters.types import (
    AdapterChatCompletion,
    AdapterChatCompletionChunk,
    AdapterFinishReason,
    Cost,
    Model,
    Provider,
    Vendor,
    ChatCompletionMessageToolCall,
    FunctionToolCall,
    Choice,
    ChatCompletionMessage,
    CompletionUsage,
    NotGiven,
    ChatCompletionCreateArgs,
    Turn,
    ConversationRole,
)

CACHE_PROMPT_TOKEN_PREMIUM = 1.25 if ADAPTERS_ENABLE_CACHE_PRICING else 0
CACHED_PROMPT_TOKEN_DISCOUNT = 0.9 if ADAPTERS_ENABLE_CACHE_PRICING else 0


class AnthropicModel(Model):
    vendor_name: str = Vendor.anthropic.value
    provider_name: str = Provider.anthropic.value

    # TODO: fix vision
    supports_vision: bool = False
    # TODO: fix tools
    supports_tools: bool = False
    # TODO: fix functions
    supports_functions: bool = False

    supports_completion: bool = False
    supports_n: bool = False

    can_system: bool = False
    can_empty_content: bool = False


MODELS: list[Model] = [
    AnthropicModel(
        name="claude-3-opus-latest",
        cost=Cost(prompt=15.00e-6, completion=75.00e-6),
        context_length=200000,
        completion_length=4096,
    ),
    AnthropicModel(
        name="claude-3-opus-20240229",
        cost=Cost(prompt=15.00e-6, completion=75.00e-6),
        context_length=200000,
        completion_length=4096,
    ),
    AnthropicModel(
        name="claude-3-5-haiku-latest",
        cost=Cost(prompt=0.80e-6, completion=4.00e-6),
        context_length=200000,
        completion_length=8192,
    ),
    AnthropicModel(
        name="claude-3-5-haiku-20241022",
        cost=Cost(prompt=0.80e-6, completion=4.00e-6),
        context_length=200000,
        completion_length=8192,
    ),
    AnthropicModel(
        name="claude-3-5-sonnet-latest",
        cost=Cost(prompt=3.00e-6, completion=15.00e-6),
        context_length=200000,
        completion_length=8192,
    ),
    AnthropicModel(
        name="claude-3-5-sonnet-20241022",
        cost=Cost(prompt=3.00e-6, completion=15.00e-6),
        context_length=200000,
        completion_length=8192,
    ),
    AnthropicModel(
        name="claude-3-5-sonnet-20240620",
        cost=Cost(prompt=3.00e-6, completion=15.00e-6),
        context_length=200000,
        completion_length=8192,
    ),
    AnthropicModel(
        name="claude-3-7-sonnet-20250219",
        cost=Cost(prompt=3.00e-6, completion=15.00e-6),
        context_length=200000,
        completion_length=8192,
    ),
    AnthropicModel(
        name="claude-3-7-sonnet-latest",
        cost=Cost(prompt=3.00e-6, completion=15.00e-6),
        context_length=200000,
        completion_length=8192,
    ),
]


class AnthropicFinishReason(str, Enum):
    end_turn = "end_turn"
    max_tokens = "max_tokens"
    stop_sequence = "stop_sequence"
    tool_use = "tool_use"


FINISH_REASON_MAPPING: Dict[AnthropicFinishReason, AdapterFinishReason] = {
    AnthropicFinishReason.end_turn: AdapterFinishReason.stop,
    AnthropicFinishReason.max_tokens: AdapterFinishReason.length,
    AnthropicFinishReason.stop_sequence: AdapterFinishReason.stop,
    AnthropicFinishReason.tool_use: AdapterFinishReason.tool_calls,
}


class AnthropicCreate(TypedDict, total=False):
    model: str
    max_tokens: int
    messages: Iterable[MessageParam]
    metadata: Optional[Metadata]
    stop_sequences: Optional[List[str]]
    stream: Optional[Literal[False] | Literal[True]]
    system: Optional[Union[str, Iterable[TextBlockParam]]]
    temperature: Optional[float]
    tool_choice: Optional[ToolChoice]
    tools: Optional[Iterable[ToolParam]]
    top_k: Optional[int]
    top_p: Optional[float]


class AnthropicSDKChatProviderAdapter(SDKChatAdapter[Anthropic, AsyncAnthropic]):
    @staticmethod
    def get_supported_models() -> list[Model]:
        return MODELS

    @staticmethod
    def get_api_key_name() -> str:
        return "ANTHROPIC_API_KEY"

    def get_base_sdk_url(self) -> str:
        return "https://api.anthropic.com"

    def _call_completion_sync(self) -> Callable[..., Any]:
        raise NotImplementedError

    def _call_completion_async(self) -> Callable[..., Any]:
        raise NotImplementedError

    def _call_sync(self) -> Callable[..., Any]:
        return self._client_sync.messages.create

    def _call_async(self) -> Callable[..., Any]:
        return self._client_async.messages.create

    def _create_client_sync(self, base_url: str, api_key: str) -> Anthropic:
        return Anthropic(
            base_url=base_url,
            api_key=api_key,
            max_retries=0,
            http_client=Client(
                limits=Limits(
                    max_connections=MAX_CONNECTIONS_PER_PROCESS,
                    max_keepalive_connections=MAX_KEEPALIVE_CONNECTIONS_PER_PROCESS,
                ),
                timeout=Timeout(timeout=HTTP_TIMEOUT, connect=HTTP_CONNECT_TIMEOUT),
            ),
        )

    def _create_client_async(self, base_url: str, api_key: str) -> AsyncAnthropic:
        return AsyncAnthropic(
            base_url=base_url,
            api_key=api_key,
            max_retries=0,
            http_client=AsyncClient(
                limits=Limits(
                    max_connections=MAX_CONNECTIONS_PER_PROCESS,
                    max_keepalive_connections=MAX_KEEPALIVE_CONNECTIONS_PER_PROCESS,
                ),
                timeout=Timeout(timeout=HTTP_TIMEOUT, connect=HTTP_CONNECT_TIMEOUT),
            ),
        )

    def _adjust_temperature(self, temperature: float) -> float:
        return temperature / 2

    def _get_params(
        self,
        *,
        stream: Optional[Literal[False]] | Literal[True] | NotGiven,
        **kwargs: Unpack[ChatCompletionCreateArgs],
    ) -> dict[str, Any]:
        openai_messages = list(kwargs["messages"])
        kwargs["messages"] = openai_messages

        system_prompt: Optional[Union[str, Iterable[TextBlockParam]]] = None

        # Extract system prompt if it's the first message, only works for str
        if (
            len(openai_messages)
            and openai_messages[0]["role"] == ConversationRole.system.value
            and isinstance(openai_messages[0]["content"], str)
        ):
            system_prompt = openai_messages[0]["content"]
            kwargs["messages"] = openai_messages[1:]

        params = super()._get_params(stream=stream, **kwargs)

        anthropic_messages = params["messages"]

        # Remove trailing whitespace from the last assistant message
        if (
            len(anthropic_messages)
            and anthropic_messages[-1]["role"] == ConversationRole.assistant.value
        ):
            if isinstance(anthropic_messages[-1]["content"], str):
                anthropic_messages[-1]["content"] = anthropic_messages[-1][
                    "content"
                ].rstrip()
            elif anthropic_messages[-1]["content"]:
                messages_content_list = list(anthropic_messages[-1]["content"])
                if messages_content_list[-1]["type"] == "text":
                    messages_content_list[-1]["text"] = messages_content_list[-1][
                        "text"
                    ].rstrip()
                anthropic_messages[-1]["content"] = messages_content_list

        # Include base64-encoded images in the request
        for message in anthropic_messages:
            if (
                isinstance(message["content"], list)
                and message["role"] == ConversationRole.user.value
            ):
                anthropic_content = []

                for content in message["content"]:
                    if content["type"] == "text":
                        anthropic_content.append(content)
                    elif content["type"] == "image_url":
                        anthropic_content.append(
                            process_image_url_anthropic(content["image_url"]["url"])
                        )

                message["content"] = anthropic_content

        # Convert tools to anthropic format
        openai_tools = params.get("tools")
        openai_tools_choice = params.get("tool_choice")

        anthropic_tools: Optional[list[ToolParam]] = None

        if openai_tools:
            anthropic_tools = []
            for openai_tool in openai_tools:
                anthropic_tool = ToolParam(
                    name=openai_tool["function"]["name"],
                    description=openai_tool["function"]["description"],
                    input_schema={
                        "type": openai_tool["function"]["parameters"]["type"],
                        "properties": openai_tool["function"]["parameters"][
                            "properties"
                        ],
                        "required": openai_tool["function"]["parameters"]["required"],
                    },
                )

                anthropic_tools.append(anthropic_tool)

        anthropic_tool_choice: Optional[ToolChoice] = None

        if openai_tools_choice == "required":
            anthropic_tool_choice = ToolChoiceToolChoiceAny(type="any")
        elif openai_tools_choice == "auto":
            anthropic_tool_choice = ToolChoiceToolChoiceAuto(type="auto")
        elif openai_tools_choice == "none":
            anthropic_tools = None
        elif isinstance(openai_tools_choice, dict):
            anthropic_tool_choice = ToolChoiceToolChoiceTool(
                name=openai_tools_choice["function"]["name"],
                type="tool",
            )

        anthropic_create: AnthropicCreate = {
            "model": self.get_model().get_api_path(),
            "max_tokens": params.get("max_tokens", self.get_model().completion_length),
            "messages": anthropic_messages,
            "metadata": params.get("metadata"),
            "stop_sequences": params.get("stop_sequences"),
            "stream": bool(params.get("stream")),
            "system": system_prompt,
            "temperature": params.get("temperature"),
            "tool_choice": anthropic_tool_choice,
            "tools": anthropic_tools,
            "top_k": params.get("top_k"),
            "top_p": params.get("top_p"),
        }

        return delete_none_values(cast(dict[str, Any], anthropic_create))

    def _extract_chat_completion_response(
        self, request: Any, response: Message
    ) -> AdapterChatCompletion:
        finish_reason = FINISH_REASON_MAPPING.get(
            AnthropicFinishReason(response.stop_reason), AdapterFinishReason.stop
        )

        choices: list[Choice] = []
        for content in response.content:
            if content.type == "text":
                choices.append(
                    Choice(
                        index=len(choices),
                        finish_reason=finish_reason.value,
                        message=ChatCompletionMessage(
                            role=ConversationRole.assistant.value,
                            content=content.text,
                        ),
                    )
                )
            elif content.type == "tool_use":
                choices.append(
                    Choice(
                        index=len(choices),
                        finish_reason=finish_reason.value,
                        message=ChatCompletionMessage(
                            role=ConversationRole.assistant.value,
                            tool_calls=[
                                ChatCompletionMessageToolCall(
                                    id=content.id,
                                    type="function",
                                    function=FunctionToolCall(
                                        name=content.name,
                                        arguments=json.dumps(content.input),
                                    ),
                                )
                            ],
                        ),
                    )
                )
            else:
                raise ValueError("Unsupported response")

        cached_creation_input_tokens = (
            response.usage.cache_creation_input_tokens
            if response.usage.cache_creation_input_tokens
            else 0
        )
        cached_read_input_tokens = (
            response.usage.cache_read_input_tokens
            if response.usage.cache_read_input_tokens
            else 0
        )

        uncached_input_tokens = response.usage.input_tokens

        cost = (
            uncached_input_tokens * self.get_model().cost.prompt
            + cached_creation_input_tokens * CACHE_PROMPT_TOKEN_PREMIUM
            + cached_read_input_tokens
            * self.get_model().cost.prompt
            * (1 - CACHED_PROMPT_TOKEN_DISCOUNT)
            + self.get_model().cost.completion * response.usage.output_tokens
            + self.get_model().cost.request
        )

        usage = CompletionUsage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )

        return AdapterChatCompletion(
            id=response.id,
            created=int(time.time()),
            model=self.get_model().name,
            object="chat.completion",
            cost=cost,
            usage=usage,
            choices=choices,
            # Deprecated
            response=Turn(
                role=ConversationRole.assistant,
                content=choices[0].message.content or "",
            ),
            token_counts=Cost(
                prompt=response.usage.input_tokens,
                completion=response.usage.output_tokens,
                request=self.get_model().cost.request,
            ),
        )

    def _extract_stream_response(
        self, request: Any, response: RawMessageStreamEvent, state: dict[str, Any]
    ) -> AdapterChatCompletionChunk:
        choice_chunk = ChoiceChunk(
            index=0,
            delta=ChoiceDelta(role=ConversationRole.assistant.value, content=""),
        )

        if isinstance(response, RawMessageStartEvent):
            state["id"] = response.message.id
            state["created"] = int(time.time())
        elif isinstance(response, RawContentBlockDeltaEvent) and isinstance(
            response.delta, TextDelta
        ):
            choice_chunk.delta.content = response.delta.text
        elif isinstance(response, RawMessageDeltaEvent) and response.delta.stop_reason:
            choice_chunk.finish_reason = FINISH_REASON_MAPPING.get(
                AnthropicFinishReason(response.delta.stop_reason),
                AdapterFinishReason.stop,
            ).value

        return AdapterChatCompletionChunk(
            id=state["id"],
            choices=[choice_chunk],
            created=state["created"],
            model=self.get_model().name,
            object="chat.completion.chunk",
        )

    def _extract_completion_response(
        self,
        request: Any,
        response: Any,
    ) -> Any:
        raise NotImplementedError

    def _extract_completion_stream_response(
        self, request: Any, response: Any, state: dict[str, Any]
    ) -> Any:
        raise NotImplementedError

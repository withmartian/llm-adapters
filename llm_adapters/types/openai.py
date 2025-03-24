# TODO: Remove this once openai types are updated
# mypy: ignore-errors

from typing import Mapping
from openai import NOT_GIVEN, NotGiven
from openai.types.completion import Completion
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
)
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.completion_create_params import (
    FunctionCall,
    Function as FunctionCreate,
    ResponseFormat,
)

from openai.types.chat.chat_completion_message_tool_call import (
    Function as FunctionToolCall,
)
from openai.types.chat.chat_completion_modality import ChatCompletionModality
from openai.types.chat.chat_completion_audio_param import ChatCompletionAudioParam
from openai.types.chat.chat_completion_prediction_content_param import (
    ChatCompletionPredictionContentParam,
)
from openai.types.chat.chat_completion_reasoning_effort import (
    ChatCompletionReasoningEffort,
)
from openai.types.chat.chat_completion_stream_options_param import (
    ChatCompletionStreamOptionsParam,
)
from openai.types.chat.chat_completion_tool_choice_option_param import (
    ChatCompletionToolChoiceOptionParam,
)
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion_chunk import Choice as ChoiceChunk, ChoiceDelta
from openai.types.chat.chat_completion import (
    Choice,
    ChatCompletionMessage,
    CompletionUsage,
)
from openai.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)
from openai.resources.files import FileObject
from openai.types.file_purpose import FilePurpose
from openai.types.shared_params.metadata import Metadata
from openai.resources.batches import Batch
from openai.pagination import SyncCursorPage, AsyncCursorPage
from openai._legacy_response import HttpxBinaryResponseContent
from openai._base_client import AsyncPaginator

Query = Mapping[str, object]
Body = object
Headers = Mapping[str, str]

ChatModel = str

__all__ = [
    "NotGiven",
    "NOT_GIVEN",
    "Completion",
    "ChatCompletion",
    "ChatCompletionChunk",
    "ChatCompletionMessageParam",
    "Query",
    "Body",
    "Headers",
    "ChatModel",
    "ChatCompletionAudioParam",
    "FunctionCall",
    "FunctionCreate",
    "FunctionToolCall",
    "ChatCompletionModality",
    "ChatCompletionPredictionContentParam",
    "ResponseFormat",
    "ChatCompletionReasoningEffort",
    "ChatCompletionStreamOptionsParam",
    "ChatCompletionToolChoiceOptionParam",
    "ChatCompletionToolParam",
    "ChatCompletionMessageToolCall",
    "Choice",
    "ChatCompletionMessage",
    "CompletionUsage",
    "ChatCompletionUserMessageParam",
    "ChoiceChunk",
    "ChoiceDelta",
    "FileObject",
    "FilePurpose",
    "Metadata",
    "Batch",
    "HttpxBinaryResponseContent",
    "SyncCursorPage",
    "AsyncCursorPage",
    "AsyncPaginator",
]

import os
import re

from llm_adapters import general_utils
from llm_adapters.abstract_adapters import (
    ApiKeyAdapterMixin,
    Adapter,
    OpenAISDKChatAdapter,
    SDKChatAdapter,
    ProviderAdapterMixin,
)
from llm_adapters.types import (
    Cost,
    Model,
    AdapterChatCompletion,
    AdapterStreamChatCompletion,
    AdapterStreamSyncChatCompletion,
    AdapterStreamAsyncChatCompletion,
    AdapterChatCompletionChunk,
    AdapterException,
    RateLimitException,
    AdapterCompletion,
    AdapterCompletionChunk,
    AdapterStreamCompletion,
    AdapterStreamSyncCompletion,
    AdapterStreamAsyncCompletion,
    Provider,
    Vendor,
    ConversationRole,
    AdapterFinishReason,
    ContentType,
    ModelProperties,
)
from llm_adapters.adapter_factory import AdapterFactory
from llm_adapters.client import OpenAI, AsyncOpenAI

# Get version from pyproject.toml
try:
    # Find the project root directory
    _root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _pyproject_path = os.path.join(_root_dir, "pyproject.toml")

    # Read the version from pyproject.toml using regex
    with open(_pyproject_path, "r", encoding="utf-8") as f:
        _content = f.read()
        _version_match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', _content)
        if _version_match:
            __version__ = _version_match.group(1)
        else:
            __version__ = "8.1.0"  # Current version as fallback
except (FileNotFoundError, Exception):
    __version__ = "unknown"  # Current version as fallback

__all__ = [
    "__version__",
    "OpenAI",
    "AsyncOpenAI",
    "general_utils",
    "ApiKeyAdapterMixin",
    "Adapter",
    "OpenAISDKChatAdapter",
    "SDKChatAdapter",
    "ProviderAdapterMixin",
    "TurnType",
    "ContentTurn",
    "FunctionOutputTurn",
    "ToolOutputTurn",
    "ToolsCallTurn",
    "FunctionCallTurn",
    "Cost",
    "Model",
    "AdapterChatCompletion",
    "AdapterStreamChatCompletion",
    "AdapterStreamSyncChatCompletion",
    "AdapterStreamAsyncChatCompletion",
    "AdapterChatCompletionChunk",
    "AdapterException",
    "RateLimitException",
    "Prompt",
    "AdapterResponse",
    "AdapterStreamResponse",
    "AdapterCompletion",
    "AdapterCompletionChunk",
    "AdapterStreamCompletion",
    "AdapterStreamSyncCompletion",
    "AdapterStreamAsyncCompletion",
    "Provider",
    "Vendor",
    "ConversationRole",
    "AdapterFinishReason",
    "VisionImageDetails",
    "TextContentEntry",
    "ImageContentEntry",
    "ContentType",
    "ModelProperties",
    "AdapterFactory",
]

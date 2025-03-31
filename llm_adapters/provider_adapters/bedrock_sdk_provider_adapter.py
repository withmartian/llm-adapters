import asyncio
from functools import partial
import json
import os
import re
import time
from typing import Any, Dict, List, Pattern, Sequence, TypeVar, cast, Literal

import boto3  # type: ignore

from llm_adapters.abstract_adapters.sdk_chat_adapter import SDKChatAdapter
from llm_adapters.types import (
    AdapterChatCompletion,
    AdapterChatCompletionChunk,
    AdapterCompletion,
    AdapterCompletionChunk,
    ChatCompletionMessage,
    CompletionUsage,
    ConversationRole,
    Cost,
    Model,
)

# Create a type alias for a Choice-like dictionary
# This is a workaround for when the actual Choice type isn't directly importable
ChoiceDict = Dict[str, Any]

PROVIDER_NAME = "bedrock-runtime"
BASE_URL = "bedrock.us-east-1.amazonaws.com"
API_SECRET_KEY_NAME = "AWS_BEDROCK_API_KEY"
API_KEY_ID = "AWS_BEDROCK_API_KEY_ID"
API_KEY_PATTERN = re.compile(r".*")
ANTHROPIC_VERSION = "bedrock-2023-05-31"
REGION_NAME = "us-east-1"
MAX_TOKENS = 1000


class BedrockModel(Model):
    # TODO: Add support for streaming and tools
    supports_streaming: bool = False
    supports_tools: bool = False

    supports_n: bool = False

    supports_functions: bool = True
    supports_json_output: bool = True
    supports_json_content: bool = True
    can_assistant_first: bool = False
    can_repeating_roles: bool = False

    can_system: bool = False
    can_system_only: bool = False
    can_system_multiple: bool = False
    can_system_last: bool = False

    can_empty_content: bool = False
    supports_completion: bool = False
    vendor_name: str = PROVIDER_NAME
    provider_name: str = PROVIDER_NAME
    modelId: str = ""


# TODO: Add more models
MODELS: Sequence[Model] = [
    BedrockModel(
        name="claude-3-haiku",
        cost=Cost(prompt=0.35e-6, completion=0.70e-6),
        context_length=128000,
        completion_length=8192,
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
    ),
    BedrockModel(
        name="claude-3-sonnet",
        cost=Cost(prompt=0.35e-6, completion=0.70e-6),
        context_length=128000,
        completion_length=8192,
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
    ),
    BedrockModel(
        name="claude-2.1",
        cost=Cost(prompt=0.35e-6, completion=0.70e-6),
        context_length=128000,
        completion_length=8192,
        modelId="anthropic.claude-v2:1",
    ),
    BedrockModel(
        name="claude-2.0",
        cost=Cost(prompt=0.35e-6, completion=0.70e-6),
        context_length=128000,
        completion_length=8192,
        modelId="anthropic.claude-v2",
    ),
    BedrockModel(
        name="claude-instant",
        cost=Cost(prompt=0.35e-6, completion=0.70e-6),
        context_length=128000,
        completion_length=8192,
        modelId="anthropic.claude-instant-v1",
    ),
]

FinishReason = Literal[
    "stop", "length", "tool_calls", "content_filter", "function_call"
]

FINISH_REASON_MAPPING: Dict[str, FinishReason] = {
    "end_turn": "stop",
    "max_tokens": "length",
}

T = TypeVar("T", bound=Model)
R = TypeVar("R")  # Response type


class BedrockSDKChatProviderAdapter(SDKChatAdapter[BedrockModel, Dict[str, Any]]):
    # Define client as Any to avoid type conflicts
    _client_sync: Any = None

    @staticmethod
    def get_supported_models() -> List[Model]:
        return list(MODELS)

    @staticmethod
    def get_provider_name() -> str:
        return PROVIDER_NAME

    def get_base_sdk_url(self) -> str:
        return BASE_URL

    @staticmethod
    def get_api_key_name() -> str:
        return API_SECRET_KEY_NAME

    @staticmethod
    def get_api_key_pattern() -> Pattern[str]:
        return API_KEY_PATTERN

    def get_api_key_id(self) -> str:
        return API_KEY_ID

    def get_anthropic_version(self) -> str:
        return ANTHROPIC_VERSION

    def get_max_tokens(self) -> int:
        return MAX_TOKENS

    def get_region_name(self) -> str:
        return REGION_NAME

    def _create_client_sync(self, base_url: str, api_key: str) -> Any:
        access_key_id = os.environ.get(self.get_api_key_id())

        return boto3.client(
            self.get_provider_name(),
            region_name=self.get_region_name(),
            aws_access_key_id=access_key_id,
            aws_secret_access_key=api_key,
        )

    def _create_client_async(self, base_url: str, api_key: str) -> Any:
        # Reuse sync client since boto3 doesn't support async natively
        return self._create_client_sync(base_url, api_key)

    def _call_sync(self) -> Any:
        if self._client_sync is None:
            raise ValueError("Sync client not initialized")

        actual_call = self._client_sync.invoke_model

        def wrapped_call(**kwargs: Any) -> Any:
            # Transform parameters to boto3 format
            boto3_params = self._transform_params_for_boto3(**kwargs)

            try:
                return actual_call(**boto3_params)
            except Exception as e:
                print(f"ERROR from Bedrock API: {str(e)}")
                raise

        return wrapped_call

    def _call_async(self) -> Any:
        if self._client_sync is None:
            raise ValueError("Sync client not initialized")

        sync_call = self._client_sync.invoke_model

        async def async_wrapper(**kwargs: Any) -> Any:
            # Transform parameters to boto3 format
            boto3_params = self._transform_params_for_boto3(**kwargs)

            loop = asyncio.get_event_loop()
            try:
                return await loop.run_in_executor(
                    None, partial(sync_call, **boto3_params)
                )
            except Exception as e:
                print(f"ERROR from Bedrock API: {str(e)}")
                raise

        return async_wrapper

    def _transform_params_for_boto3(self, **kwargs: Any) -> Dict[str, Any]:
        """Transform SDK parameters to boto3 format"""
        model = cast(BedrockModel, self.get_model())
        model_id = model.modelId

        # Extract the exact parameters boto3 expects
        return {
            "modelId": model_id,
            "body": json.dumps(self._format_request_body(**kwargs)),
            "contentType": "application/json",
            "accept": "application/json",
        }

    def _format_request_body(self, **kwargs: Any) -> Dict[str, Any]:
        """Format the request body for Claude models"""
        messages = kwargs.get("messages", [])
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", self.get_max_tokens())

        # If first message is from assistant, prepend a user message
        if messages and messages[0].get("role") == "assistant":
            messages = [{"role": "user", "content": "Hello"}] + messages

        # Clean up any trailing whitespace in messages
        cleaned_messages = []
        for message in messages:
            cleaned_message = message.copy()

            if isinstance(cleaned_message.get("content"), str):
                # Remove trailing whitespace from string content
                cleaned_message["content"] = cleaned_message["content"].rstrip()
            elif isinstance(cleaned_message.get("content"), list):
                cleaned_content = []
                for item in cleaned_message["content"]:
                    if isinstance(item, dict) and "text" in item:
                        item_copy = item.copy()
                        item_copy["text"] = item_copy["text"].rstrip()
                        cleaned_content.append(item_copy)
                    else:
                        cleaned_content.append(item)
                cleaned_message["content"] = cleaned_content

            cleaned_messages.append(cleaned_message)

        request_body = {
            "anthropic_version": self.get_anthropic_version(),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": cleaned_messages,
        }

        return request_body

    def _call_completion_sync(self) -> Any:
        # Bedrock doesn't support the completion API separately
        raise NotImplementedError("Completion API not supported for Bedrock")

    def _call_completion_async(self) -> Any:
        # Bedrock doesn't support the completion API separately
        raise NotImplementedError("Completion API not supported for Bedrock")

    def _extract_response(self, request: Any, response: Any) -> AdapterChatCompletion:
        body = json.loads(response["body"].read())

        content = None

        if "content" in body and isinstance(body["content"], list):
            text_items = [item for item in body["content"] if item["type"] == "text"]
            if text_items:
                content = text_items[0].get("text", "")

        finish_reason = FINISH_REASON_MAPPING.get(body.get("stop_reason", ""), "stop")

        prompt_tokens = body.get("usage", {}).get("input_tokens", 0)
        completion_tokens = body.get("usage", {}).get("output_tokens", 0)

        model = self.get_model()
        cost = (
            model.cost.prompt * prompt_tokens
            + model.cost.completion * completion_tokens
            + model.cost.request
        )

        # Convert date string to timestamp
        created_str = (
            response.get("ResponseMetadata", {}).get("HTTPHeaders", {}).get("date", "")
        )
        created = 0
        if created_str:
            try:
                # Convert RFC 1123 date format to timestamp
                from email.utils import parsedate_to_datetime

                created = int(parsedate_to_datetime(created_str).timestamp())
            except Exception:
                # Fallback to current time if parsing fails
                created = int(time.time())
        else:
            created = int(time.time())

        message = ChatCompletionMessage(
            role="assistant",
            content=content,
        )

        usage = CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

        choice: ChoiceDict = {
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }

        return AdapterChatCompletion(
            id=response.get("ResponseMetadata", {}).get("RequestId", ""),
            object="chat.completion",
            choices=[choice],  # type: ignore
            created=created,
            model=model.name,
            usage=usage,
            cost=cost,
        )

    def _extract_stream_response(
        self, request: Any, response: Any, state: Dict[str, Any]
    ) -> AdapterChatCompletionChunk:
        if not state:
            state["finish_reason"] = None
            state["content"] = ""
            state["role"] = ConversationRole.assistant.value

        # TODO: Implement streaming for Claude models
        # Placeholder for streaming implementation
        content = None

        # Create a delta message
        delta = ChatCompletionMessage(
            role=state["role"],
            content=content,
        )

        chunk_choice: ChoiceDict = {
            "index": 0,
            "delta": delta,
            "finish_reason": state["finish_reason"],
        }

        return AdapterChatCompletionChunk(
            id="placeholder",
            object="chat.completion.chunk",
            choices=[chunk_choice],  # type: ignore
            created=int(time.time()),
            model=self.get_model().name,
        )

    def _extract_completion_response(
        self, request: Any, response: Any
    ) -> AdapterCompletion:
        # Not supported for Claude models
        raise NotImplementedError("Completion API not supported for Bedrock")

    def _extract_completion_stream_response(
        self, request: Any, response: Any, state: Dict[str, Any]
    ) -> AdapterCompletionChunk:
        # Not supported for Claude models
        raise NotImplementedError("Completion API not supported for Bedrock")

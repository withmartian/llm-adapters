import json
import re
from typing import Any, Dict, Pattern

import boto3

from llm_adapters.abstract_adapters.api_key_adapter_mixin import ApiKeyAdapterMixin
from llm_adapters.abstract_adapters.provider_adapter_mixin import ProviderAdapterMixin
from llm_adapters.abstract_adapters.sdk_chat_adapter import SDKChatAdapter
from llm_adapters.types import (
    AdapterChatCompletion,
    AdapterChatCompletionChunk,
    AdapterCompletion,
    AdapterCompletionChunk,
    AdapterException,
    ConversationRole,
    Model,
    Cost
)

PROVIDER_NAME = "bedrock-runtime"
BASE_URL = "bedrock.us-east-1.amazonaws.com"
API_SECRET_KEY_NAME = "AWS_BEDROCK_API_KEY"
API_KEY_ID = "AWS_BEDROCK_API_KEY_ID"
API_KEY_PATTERN = re.compile(r".*")
ANTHROPIC_VERSION = "bedrock-2023-05-31"
REGION_NAME = "us-east-1"
MAX_TOKENS = 1000


class BedrockModel(Model):
    supports_streaming: bool = True
    supports_functions: bool = True
    supports_tools: bool = True
    supports_n: bool = True
    supports_json_output: bool = True
    supports_json_content: bool = True
    supports_first_assistant: bool = False
    supports_system: bool = False
    supports_empty_content: bool = False
    vendor_name: str = PROVIDER_NAME
    provider_name: str = PROVIDER_NAME


MODELS = [
    BedrockModel(
        name="jamba-instruct",
        cost=Cost(prompt=0.125e-6, completion=0.375e-6),
        context_length=30720,
        completion_length=2048,
        modelId="ai21.jamba-instruct-v1:0",
    ),
    BedrockModel(
        name="jurassic-2-ultra",
        cost=Cost(prompt=0.125e-6, completion=0.375e-6),
        context_length=30720,
        completion_length=2048,
        modelId="ai21.j2-ultra-v1",
    ),
    BedrockModel(
        name="jurassic-2-mid",
        cost=Cost(prompt=3.5e-6, completion=10.5e-6),
        context_length=128000,
        completion_length=8192,
        modelId="ai21.j2-mid-v1",
    ),
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
    BedrockModel(
        name="command-r-plus",
        cost=Cost(prompt=0.3e-6, completion=1.5e-6),
        context_length=128000,
        completion_length=8192,
        modelId="cohere.command-r-plus-v1:0",
    ),
    BedrockModel(
        name="command-r",
        cost=Cost(prompt=0.05e-6, completion=0.15e-6),
        context_length=128000,
        completion_length=8192,
        modelId="cohere.command-r-v1:0",
    ),
    BedrockModel(
        name="llama3-8b-instruct",
        cost=Cost(prompt=0.03e-6, completion=0.06e-6),
        context_length=8000,
        completion_length=8192,
        modelId="meta.llama3-8b-instruct-v1:0",
    ),
    BedrockModel(
        name="llama3-70b-instruct",
        cost=Cost(prompt=0.265e-6, completion=0.35e-6),
        context_length=8000,
        completion_length=8219,
        modelId="meta.llama3-70b-instruct-v1:0",
    ),
    BedrockModel(
        name="mistral-7b-instruct",
        cost=Cost(prompt=0.15e-6, completion=0.2e-6),
        context_length=32000,
        completion_length=8192,
        modelId="mistral.mistral-7b-instruct-v0:2",
    ),
    BedrockModel(
        name="mixtral-8x7b-instruct",
        cost=Cost(prompt=0.45e-6, completion=0.7e-6),
        context_length=32000,
        completion_length=8192,
        modelId="mistral.mixtral-8x7b-instruct-v0:1",
    ),
    BedrockModel(
        name="mistral-large-2402",
        cost=Cost(prompt=4e-6, completion=12e-6),
        context_length=32000,
        completion_length=8192,
        modelId="mistral.mistral-large-2402-v1:0",
    ),
    BedrockModel(
        name="mistral-small-2402",
        cost=Cost(prompt=1e-6, completion=3e-6),
        context_length=32000,
        completion_length=8192,
        modelId="mistral.mistral-small-2402-v1:0",
    ),
]


FINISH_REASON_MAPPING = {
    "end_turn": "stop",
    "max_tokens": "length",
}


FINISH_REASON_MAPPING = {
    "end_turn": "stop",
    "max_tokens": "length",
}

class BedrockSDKChatProviderAdapter(SDKChatAdapter):
    @staticmethod
    def get_supported_models():
        return MODELS

    @staticmethod
    def get_provider_name() -> str:
        return PROVIDER_NAME

    def get_base_sdk_url(self) -> str:
        return BASE_URL

    @staticmethod
    def get_api_key_name() -> str:
        return API_SECRET_KEY_NAME

    @staticmethod
    def get_api_key_pattern() -> Pattern:
        return API_KEY_PATTERN
    
    def get_api_key_id(self) -> str:
        return API_KEY_ID

    def get_anthropic_version(self) -> str:
        return ANTHROPIC_VERSION

    def get_max_tokens(self) -> int:
        return MAX_TOKENS

    def get_region_name(self) -> str:
        return REGION_NAME

    def _create_client_sync(self, base_url: str, api_key: str):
        return boto3.client(
            self.get_provider_name(),
            region_name=self.get_region_name(),
            aws_access_key_id=self.get_api_key_id(),
            aws_secret_access_key=api_key,
        )
    
    def _create_client_async(self, base_url: str, api_key: str):
        # Reuse sync client since boto3 doesn't support async natively
        return self._create_client_sync(base_url, api_key)

    def _call_sync(self):
        return self._client_sync.invoke_model
    
    def _call_async(self):
        sync_call = self._call_sync()
        
        async def async_wrapper(**kwargs):
            loop = asyncio.get_event_loop()
            if kwargs.get('stream', False):
                # For streaming, we need to handle it differently
                response = await loop.run_in_executor(
                    None, 
                    partial(sync_call, **kwargs)
                )
                
                # Create an async generator to yield chunks
                async def stream_generator():
                    try:
                        for chunk in response.get('body', []):
                            yield chunk
                            await asyncio.sleep(0)
                    finally:
                        if hasattr(response.get('body', None), 'close'):
                            response['body'].close()
                
                return stream_generator()
            else:
                # For non-streaming responses
                return await loop.run_in_executor(
                    None, 
                    partial(sync_call, **kwargs)
                )
        
        return async_wrapper
    
    def _call_completion_sync(self):
        # Bedrock doesn't support the completion API separately
        raise NotImplementedError("Completion API not supported for Bedrock")
    
    def _call_completion_async(self):
        # Bedrock doesn't support the completion API separately
        raise NotImplementedError("Completion API not supported for Bedrock")
    
    # Response extraction methods
    def _extract_response(self, request: Any, response: Any) -> AdapterChatCompletion:
        # Extract from the existing logic
        body = json.loads(response["body"].read())
        
        choices = [
            {
                "message": {
                    "role": body["role"],
                    "content": choice["text"],
                },
                "finish_reason": FINISH_REASON_MAPPING.get(
                    body["stop_reason"], body["stop_reason"]
                ),
            }
            for choice in body["content"]
        ]
        
        prompt_tokens = body["usage"]["input_tokens"]
        completion_tokens = body["usage"]["output_tokens"]
        
        model = self.get_model()
        cost = (
            model.cost.prompt * prompt_tokens
            + model.cost.completion * completion_tokens
            + model.cost.request
        )
        
        return AdapterChatCompletion(
            id=response.get("ResponseMetadata", {}).get("RequestId", ""),
            choices=choices,
            created=response.get("ResponseMetadata", {}).get("HTTPHeaders", {}).get("date", 0),
            model=model.name,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            },
            cost=cost,
        )
    
    def _extract_stream_response(
        self, request: Any, response: Any, state: Dict[str, Any]
    ) -> AdapterChatCompletionChunk:
        # New method for handling streaming responses
        content = None
        
        # Initialize state if needed
        if not state:
            state["finish_reason"] = None
            state["content"] = ""
            state["role"] = ConversationRole.assistant.value
        
        chunk_type = getattr(response, "type", None)
        delta = getattr(response, "delta", None)
        
        if delta and hasattr(delta, "text"):
            content = delta.text
            state["content"] += content
        
        if chunk_type == "message_stop":
            state["finish_reason"] = "stop"
        
        return AdapterChatCompletionChunk(
            id=getattr(response, "id", ""),
            choices=[
                {
                    "index": 0,
                    "delta": {
                        "role": state["role"],
                        "content": content,
                    },
                    "finish_reason": state["finish_reason"],
                }
            ],
            created=0,  # Bedrock doesn't provide this in stream chunks
            model=self.get_model().name,
        )
    
    def _extract_completion_response(self, request: Any, response: Any) -> AdapterCompletion:
        # Bedrock doesn't support the completion API separately
        raise NotImplementedError("Completion API not supported for Bedrock")
    
    def _extract_completion_stream_response(
        self, request: Any, response: Any, state: Dict[str, Any]
    ) -> AdapterCompletionChunk:
        # Bedrock doesn't support the completion API separately
        raise NotImplementedError("Completion API not supported for Bedrock")
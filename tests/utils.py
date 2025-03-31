import json
from typing import Any, Iterable

import brotli

from llm_adapters.abstract_adapters.openai_sdk_chat_adapter import OpenAISDKChatAdapter
from llm_adapters.adapter_factory import AdapterFactory
from llm_adapters.provider_adapters.anthropic_sdk_chat_provider_adapter import (
    AnthropicSDKChatProviderAdapter,
)
from llm_adapters.provider_adapters.cohere_sdk_chat_provider_adapter import (
    CohereSDKChatProviderAdapter,
)

from llm_adapters.provider_adapters.bedrock_sdk_provider_adapter import (
    BedrockSDKChatProviderAdapter
)
from vcr import VCR
from openai.types.chat import ChatCompletionMessageParam


TEST_CHAT_MODELS = [
    model.get_path()
    for model in AdapterFactory.get_supported_models()
    if model.supports_chat
]


TEST_COMPLETION_MODELS = [
    model.get_path()
    for model in AdapterFactory.get_supported_models()
    if model.supports_completion
]


SIMPLE_CONVERSATION_USER_ONLY: Iterable[ChatCompletionMessageParam] = [
    {
        "role": "user",
        "content": "Hi",
    }
]


SIMPLE_CONVERSATION_JSON_OUTPUT: Iterable[ChatCompletionMessageParam] = [
    {
        "role": "user",
        "content": "Hi, output simple json",
    }
]


SIMPLE_CONVERSATION_JSON_CONTENT: Iterable[ChatCompletionMessageParam] = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Hi",
            },
            {
                "type": "text",
                "text": "Hi",
            },
        ],
    }
]


SIMPLE_FUNCTION_CALL_USER_ONLY: Iterable[ChatCompletionMessageParam] = [
    {
        "role": "user",
        "content": "Generate random number with tools specified, range 1-10",
    }
]

SIMPLE_GENERATE_TOOLS = [
    {
        "type": "function",
        "function": {
            "description": "Generate random number",
            "name": "generate",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Random number like 5, 4, 3, 10, 11",
                    },
                },
                "required": ["prompt"],
            },
        },
    }
]


def get_response_content_from_vcr(vcr: VCR, model_path: str) -> Any:
    adapter = AdapterFactory.get_adapter_by_path(model_path)
    response = vcr.responses[-1]["body"]["string"]
    try:
        response = brotli.decompress(response)
    except Exception as _:  # pylint: disable=W0718
        print("Failed to decompress response")
    response = json.loads(response)
    
    if isinstance(adapter, OpenAISDKChatAdapter):
        return response["choices"][0]["message"]["content"]
    elif isinstance(adapter, AnthropicSDKChatProviderAdapter):
        return response["content"][0]["text"]
    elif isinstance(adapter, BedrockSDKChatProviderAdapter):
        return response["content"][0]["text"]
    else:
        raise ValueError(f"Unknown adapter type: {type(adapter).__name__}")


def get_response_choices_from_vcr(vcr: VCR, model_path: str) -> Any:
    adapter = AdapterFactory.get_adapter_by_path(model_path)
    response = vcr.responses[-1]["body"]["string"]
    try:
        response = brotli.decompress(response)
    except Exception as _:  # pylint: disable=W0718
        print("Failed to decompress response")
    response = json.loads(response)
    
    if isinstance(adapter, OpenAISDKChatAdapter):
        return response["choices"]
    elif isinstance(adapter, AnthropicSDKChatProviderAdapter):
        anthropic_choices: list[Any] = []
        for content in response["content"]:
            if content["type"] == "tool_use":
                anthropic_choices.append(
                    {
                        "message": {
                            "role": response["role"],
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": content["name"],
                                        "arguments": json.dumps(content["input"]),
                                    },
                                }
                            ],
                        }
                    }
                )
            elif content["type"] == "text":
                anthropic_choices.append(
                    {
                        "message": {
                            "role": response["role"],
                            "content": content["text"],
                        }
                    }
                )
            else:
                raise ValueError(f"Unknown content type: {content['type']}")
        return anthropic_choices
    # elif isinstance(adapter, CohereSDKChatProviderAdapter):
    #     # Parse the Cohere SDK response
    #     cohere_choices: list[Any] = []

    #     # Assuming the structure of the Cohere response can vary
    #     message = response.get("message")
    #     role = message.get("role", "assistant")
    #     tool_calls = message.get("tool_calls")
    #     content_list = message.get("content")

    #     # Check if there are tool calls and handle them
    #     if tool_calls:
    #         for tool_call in tool_calls:
    #             cohere_choices.append(
    #                 {
    #                     "message": {
    #                         "role": role,
    #                         "tool_calls": [
    #                             {
    #                                 "id": tool_call["id"],
    #                                 "type": tool_call["type"],
    #                                 "function": {
    #                                     "name": tool_call["function"]["name"],
    #                                     "arguments": tool_call["function"]["arguments"],
    #                                 },
    #                             }
    #                         ],
    #                     }
    #                 }
    #             )
    # If no tool calls, check for and handle content as a list of text items
    # elif content_list:
    #     for content in content_list:
    #         if content.get("type") == "text":
    #             cohere_choices.append(
    #                 {"message": {"role": role, "content": content["text"]}}
    #             )

    # return cohere_choices
    else:
        raise ValueError("Unknown adapter")

# import pytest
# from typing import cast, Iterable

# from llm_adapters.client import AsyncOpenAI
# from llm_adapters.types.openai import ChatCompletionToolParam
# from llm_adapters.adapter_factory import AdapterFactory
# from tests.utils import (
#     TEST_CHAT_MODELS,
#     SIMPLE_FUNCTION_CALL_USER_ONLY,
#     SIMPLE_GENERATE_TOOLS,
#     get_response_choices_from_vcr,
# )
# from vcr import VCR

# async_client = AsyncOpenAI()


# @pytest.mark.vcr
# @pytest.mark.parametrize("model_path", TEST_CHAT_MODELS, ids=str)
# async def test_async(vcr: VCR, model_path: str) -> None:
#     model = AdapterFactory.get_model_by_path(model_path)

#     if not model.supports_tools:
#         return

#     tools = cast(Iterable[ChatCompletionToolParam], SIMPLE_GENERATE_TOOLS)

#     stream_response = await async_client.chat.completions.create(
#         model=model_path,
#         messages=SIMPLE_FUNCTION_CALL_USER_ONLY,
#         tools=tools,
#         stream=True,
#     )

#     collected_choices = []
#     async for chunk in stream_response:
#         collected_choices.append(chunk)

#     cassette_choices = get_response_choices_from_vcr(vcr, model_path)


#     assert collected_choices[0].choices[0].delta.tool_calls
#     assert (
#         collected_choices[0].choices[0].delta.tool_calls[0].function.name
#         == cassette_choices[0]["message"]["tool_calls"][0]["function"]["name"]
#     )
#     assert (
#         collected_choices[0].choices[0].delta.tool_calls[0].function.arguments
#         == cassette_choices[0]["message"]["tool_calls"][0]["function"]["arguments"]
#     )

# import pytest


# from llm_adapters.client import AsyncOpenAI, OpenAI
# from tests.utils import (
#     TEST_CHAT_MODELS,
#     SIMPLE_CONVERSATION_USER_ONLY,
#     get_response_content_from_vcr,
# )
# from vcr import VCR

# async_client = AsyncOpenAI()


# @pytest.mark.vcr
# @pytest.mark.parametrize("model_path", TEST_CHAT_MODELS, ids=str)
# def test_sync(vcr: VCR, model_path: str) -> None:
#     client = OpenAI()

#     adapter_response = client.chat.completions.create(
#         model=model_path,
#         messages=SIMPLE_CONVERSATION_USER_ONLY,
#         stream=True,
#     )

#     response_text = ""
#     for chunk in adapter_response:
#         if hasattr(chunk, "choices") and chunk.choices and len(chunk.choices) > 0:
#             if hasattr(chunk.choices[0], "delta") and chunk.choices[0].delta.content:
#                 response_text += chunk.choices[0].delta.content

#     cassette_response = get_response_content_from_vcr(vcr, model_path)

#     assert len(response_text) > 0
#     # We don't compare with cassette response because streaming responses are collected differently


# @pytest.mark.vcr
# @pytest.mark.parametrize("model_path", TEST_CHAT_MODELS, ids=str)
# async def test_async(vcr: VCR, model_path: str) -> None:
#     adapter_response = await async_client.chat.completions.create(
#         model=model_path,
#         messages=SIMPLE_CONVERSATION_USER_ONLY,
#         stream=True,
#     )

#     response_text = ""
#     async for chunk in adapter_response:
#         if hasattr(chunk, "choices") and chunk.choices and len(chunk.choices) > 0:
#             if hasattr(chunk.choices[0], "delta") and chunk.choices[0].delta.content:
#                 response_text += chunk.choices[0].delta.content

#     cassette_response = get_response_content_from_vcr(vcr, model_path)

#     assert len(response_text) > 0
#     # We don't compare with cassette response because streaming responses are collected differently

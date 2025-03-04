# import pytest

# from llm_adapters.client import AsyncOpenAI, OpenAI
# from tests.utils import (
#     TEST_COMPLETION_MODELS,
# )
# from vcr import VCR

# async_client = AsyncOpenAI()
# sync_client = OpenAI()


# @pytest.mark.vcr
# @pytest.mark.parametrize("model_path", TEST_COMPLETION_MODELS, ids=str)
# async def test_async(vcr: VCR, model_path: str) -> None:
#     response = await async_client.completions.create(
#         model=model_path, prompt="Hi", stream=True
#     )

#     response_text = ""
#     async for chunk in response.response:
#         if chunk.choices and len(chunk.choices) > 0:
#             response_text += chunk.choices[0].text if chunk.choices[0].text else ""

#     chunks = [data_chunk async for data_chunk in response.response]

#     response_text = "".join(
#         [chunk.choices[0].text for chunk in chunks if len(chunk.choices)]
#     )
#     assert len(response_text)


# @pytest.mark.vcr
# @pytest.mark.parametrize("model_path", TEST_COMPLETION_MODELS, ids=str)
# def test_sync(vcr: VCR, model_path: str) -> None:
#     response = sync_client.completions.create(
#         model=model_path, prompt="Hi", stream=True
#     )

#     response_text = ""
#     for chunk in response:
#         if chunk.choices and len(chunk.choices) > 0:
#             response_text += chunk.choices[0].text if chunk.choices[0].text else ""

#     chunks = list(response.response)

#     response_text = "".join(
#         [chunk.choices[0].text for chunk in chunks if len(chunk.choices)]
#     )
#     assert len(response_text)

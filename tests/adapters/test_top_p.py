import pytest

from llm_adapters.client import AsyncOpenAI
from tests.utils import (
    TEST_CHAT_MODELS,
    TEST_COMPLETION_MODELS,
    SIMPLE_CONVERSATION_USER_ONLY,
    get_response_content_from_vcr,
)
from vcr import VCR

async_client = AsyncOpenAI()


@pytest.mark.vcr
@pytest.mark.parametrize("model_path", TEST_CHAT_MODELS, ids=str)
async def test_async(vcr: VCR, model_path: str) -> None:
    adapter_response = await async_client.chat.completions.create(
        model=model_path, messages=SIMPLE_CONVERSATION_USER_ONLY, top_p=0.5
    )

    cassette_response = get_response_content_from_vcr(vcr, model_path)

    assert adapter_response.choices[0].message.content == cassette_response


@pytest.mark.vcr
@pytest.mark.parametrize("model_path", TEST_COMPLETION_MODELS, ids=str)
async def test_async_completion(vcr: VCR, model_path: str) -> None:
    adapter_response = await async_client.completions.create(
        model=model_path, prompt="Hi", top_p=0.5
    )

    cassette_response = get_response_content_from_vcr(vcr, model_path)

    assert adapter_response.choices[0].text == cassette_response

import pytest

from llm_adapters.client import AsyncOpenAI
from tests.utils import (
    TEST_CHAT_MODELS,
    get_response_content_from_vcr,
)
from vcr import VCR

async_client = AsyncOpenAI()


@pytest.mark.vcr
@pytest.mark.parametrize("model_path", TEST_CHAT_MODELS, ids=str)
async def test_async(vcr: VCR, model_path: str) -> None:
    adapter_response = await async_client.chat.completions.create(
        model=model_path,
        messages=[
            {
                "role": "user",
                "content": "Hi",
            },
            {
                "role": "user",
                "content": "Hi",
            },
        ],
    )

    cassette_response = get_response_content_from_vcr(vcr, model_path)

    assert adapter_response.choices[0].message.content == cassette_response

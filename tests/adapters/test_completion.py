import pytest

from llm_adapters.client import AsyncOpenAI, OpenAI
from tests.utils import (
    TEST_COMPLETION_MODELS,
)
from vcr import VCR

async_client = AsyncOpenAI()
sync_client = OpenAI()


@pytest.mark.vcr
@pytest.mark.parametrize("model_path", TEST_COMPLETION_MODELS, ids=str)
async def test_async(vcr: VCR, model_path: str) -> None:
    response = await async_client.completions.create(
        model=model_path,
        prompt="Hi",
    )

    assert response.choices[0].text

    # TODO: Fix this
    # cassette_response = get_response_content_from_vcr(vcr, model_path)
    # assert response.choices[0].text == cassette_response


@pytest.mark.vcr
@pytest.mark.parametrize("model_path", TEST_COMPLETION_MODELS, ids=str)
def test_sync(vcr: VCR, model_path: str) -> None:
    response = sync_client.completions.create(
        model=model_path,
        prompt="Hi",
    )

    assert response.choices[0].text

    # TODO: Fix this
    # cassette_response = get_response_content_from_vcr(vcr, model_path)
    # assert response.choices[0].text == cassette_response

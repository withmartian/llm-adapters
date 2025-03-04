import pytest

from llm_adapters.adapter_factory import AdapterFactory
from llm_adapters.client import AsyncOpenAI
from tests.utils import (
    TEST_CHAT_MODELS,
    SIMPLE_CONVERSATION_JSON_OUTPUT,
    get_response_content_from_vcr,
)
from vcr import VCR

async_client = AsyncOpenAI()


@pytest.mark.vcr
@pytest.mark.parametrize("model_path", TEST_CHAT_MODELS, ids=str)
async def test_async(vcr: VCR, model_path: str) -> None:
    if not AdapterFactory.get_model_by_path(model_path).supports_json_output:
        return

    response = await async_client.chat.completions.create(
        model=model_path,
        messages=SIMPLE_CONVERSATION_JSON_OUTPUT,
        response_format={"type": "json_object"},
    )

    cassette_response = get_response_content_from_vcr(vcr, model_path)

    assert response.choices[0].message.content == cassette_response

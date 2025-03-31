import pytest
from llm_adapters.adapter_factory import AdapterFactory
from llm_adapters.client import AsyncOpenAI
from tests.utils import (
    TEST_CHAT_MODELS,
    SIMPLE_CONVERSATION_USER_ONLY,
    get_response_content_from_vcr,
)
from vcr import VCR

async_client = AsyncOpenAI()

# Filter model paths to only include those that support the 'n' parameter
MODELS_SUPPORTING_N = [
    model_path
    for model_path in TEST_CHAT_MODELS
    if (model := AdapterFactory.get_model_by_path(model_path)) and model.supports_n
]


@pytest.mark.vcr
@pytest.mark.parametrize("model_path", MODELS_SUPPORTING_N, ids=str)
async def test_async(vcr: VCR, model_path: str) -> None:
    n = 2

    adapter_response = await async_client.chat.completions.create(
        model=model_path, messages=SIMPLE_CONVERSATION_USER_ONLY, n=n
    )

    cassette_response = get_response_content_from_vcr(vcr, model_path)
    assert adapter_response.choices[0].message.content == cassette_response
    assert len(adapter_response.choices) == n

import pytest
from llm_adapters.client import AsyncOpenAI
from llm_adapters.adapter_factory import AdapterFactory
from tests.utils import (
    TEST_CHAT_MODELS,
    get_response_content_from_vcr,
)
from vcr import VCR

async_client = AsyncOpenAI()

# Filter model paths to only include those that support multiple system messages
MULTI_SYSTEM_MODELS = [
    model_path for model_path in TEST_CHAT_MODELS
    if (model := AdapterFactory.get_model_by_path(model_path)) and 
       model.supports_chat and model.can_system and model.can_system_multiple
]

@pytest.mark.vcr
@pytest.mark.parametrize("model_path", MULTI_SYSTEM_MODELS, ids=str)
async def test_async(vcr: VCR, model_path: str) -> None:

    response = await async_client.chat.completions.create(
        model=model_path,
        messages=[
            {"role": "system", "content": "Hi"},
            {"role": "system", "content": "Hi"},
            {"role": "user", "content": "Hi"},
            {"role": "user", "content": "Hi"},
            {"role": "system", "content": "Hi"},
            {"role": "system", "content": "Hi"},
            {"role": "user", "content": "Hi"},
        ],
    )
    cassette_response = get_response_content_from_vcr(vcr, model_path)
    assert response.choices[0].message.content == cassette_response
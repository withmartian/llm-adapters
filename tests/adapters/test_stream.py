import pytest


from llm_adapters.adapter_factory import AdapterFactory
from llm_adapters.client import AsyncOpenAI, OpenAI
from tests.utils import (
    TEST_CHAT_MODELS,
    SIMPLE_CONVERSATION_USER_ONLY,
)
from vcr import VCR

sync_client = OpenAI()
async_client = AsyncOpenAI()


@pytest.mark.vcr
@pytest.mark.parametrize("model_path", TEST_CHAT_MODELS, ids=str)
def test_sync(vcr: VCR, model_path: str) -> None:
    model = AdapterFactory.get_model_by_path(model_path)

    if not model.supports_streaming:
        return

    response = sync_client.chat.completions.create(
        model=model_path,
        messages=SIMPLE_CONVERSATION_USER_ONLY,
        stream=True,
    )

    response_text = ""
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content is not None:
            response_text += chunk.choices[0].delta.content

    assert len(response_text) > 0
    # We don't compare with cassette response because streaming responses are collected differently
    # cassette_response = get_response_content_from_vcr(vcr, model_path)


@pytest.mark.vcr
@pytest.mark.parametrize("model_path", TEST_CHAT_MODELS, ids=str)
async def test_async(vcr: VCR, model_path: str) -> None:
    model = AdapterFactory.get_model_by_path(model_path)

    if not model.supports_streaming:
        return

    response = await async_client.chat.completions.create(
        model=model_path,
        messages=SIMPLE_CONVERSATION_USER_ONLY,
        stream=True,
    )

    response_text = ""
    async for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content is not None:
            response_text += chunk.choices[0].delta.content

    assert len(response_text) > 0
    # We don't compare with cassette response because streaming responses are collected differently
    # cassette_response = get_response_content_from_vcr(vcr, model_path)

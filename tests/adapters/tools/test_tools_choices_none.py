import pytest
from typing import cast, Iterable

from llm_adapters.adapter_factory import AdapterFactory
from llm_adapters.client import AsyncOpenAI
from llm_adapters.types.openai import ChatCompletionToolParam
from tests.utils import (
    TEST_CHAT_MODELS,
    SIMPLE_FUNCTION_CALL_USER_ONLY,
    SIMPLE_GENERATE_TOOLS,
    get_response_choices_from_vcr,
)
from vcr import VCR

async_client = AsyncOpenAI()


@pytest.mark.vcr
@pytest.mark.parametrize("model_path", TEST_CHAT_MODELS, ids=str)
async def test_async(vcr: VCR, model_path: str) -> None:
    model = AdapterFactory.get_model_by_path(model_path)

    if not model.supports_tools:
        return

    tools = cast(Iterable[ChatCompletionToolParam], SIMPLE_GENERATE_TOOLS)

    adapter_response = await async_client.chat.completions.create(
        model=model_path,
        messages=SIMPLE_FUNCTION_CALL_USER_ONLY,
        tools=tools,
        tool_choice="none",
    )

    cassette_choices = get_response_choices_from_vcr(vcr, model_path)

    assert adapter_response.choices[-1].message.content == cassette_choices[-1][
        "message"
    ].get("content", None)
    assert adapter_response.choices[-1].message.role == cassette_choices[-1][
        "message"
    ].get("role", None)

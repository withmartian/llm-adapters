import pytest

from tests.utils import (
    ADAPTER_CHAT_TEST_FACTORIES,
    SIMPLE_CONVERSATION_USER_ONLY,
    AdapterTestFactory,
    get_response_content_from_vcr,
)
from vcr import VCR


@pytest.mark.vcr
@pytest.mark.parametrize("create_adapter", ADAPTER_CHAT_TEST_FACTORIES, ids=str)
async def test_async(vcr: VCR, create_adapter: AdapterTestFactory) -> None:
    print(ADAPTER_CHAT_TEST_FACTORIES)
    adapter = create_adapter()

    adapter_response = await adapter.execute_chat_completion_async(
        messages=SIMPLE_CONVERSATION_USER_ONLY
    )

    cassette_response = get_response_content_from_vcr(vcr, adapter)

    assert adapter_response.choices[0].message.content == cassette_response

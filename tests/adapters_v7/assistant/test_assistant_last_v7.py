import pytest

from adapters.types import Conversation, ConversationRole, Turn
from tests.utils import (
    ADAPTER_CHAT_TEST_FACTORIES,
    AdapterTestFactory,
    get_response_content_from_vcr,
)
from vcr import VCR


conversation = Conversation(
    [
        Turn(role=ConversationRole.user, content="Hi"),
        Turn(role=ConversationRole.assistant, content="Hi "),
    ]
)


@pytest.mark.vcr
@pytest.mark.parametrize("create_adapter", ADAPTER_CHAT_TEST_FACTORIES, ids=str)
async def test_async(vcr: VCR, create_adapter: AdapterTestFactory) -> None:
    adapter = create_adapter()

    # Empty space intentionally added, to test trailing whitespace as some providers don't like it
    adapter_response = await adapter.execute_async(
        [
            {
                "role": "user",
                "content": "Hi",
            },
            {
                "role": "assistant",
                "content": "Hi ",
            },
        ]
    )

    cassette_response = get_response_content_from_vcr(
        vcr, adapter.get_model().get_path()
    )

    assert adapter_response.choices[0].message.content == cassette_response

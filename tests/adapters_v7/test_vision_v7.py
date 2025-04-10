# import pytest

# from tests.utils import (
#     ADAPTER_CHAT_TEST_FACTORIES,
#     AdapterTestFactory,
#     get_response_content_from_vcr,
# )
# from vcr import VCR


# @pytest.mark.vcr
# @pytest.mark.parametrize("create_adapter", ADAPTER_CHAT_TEST_FACTORIES, ids=str)
# async def test_async(vcr: VCR, create_adapter: AdapterTestFactory) -> None:
#     adapter = create_adapter()

#     if not adapter.get_model().supports_vision:
#         return

#     adapter_response = await adapter.execute_async(
#         [
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "text",
#                         "text": "Describe the image",
#                     },
#                     {
#                         # Elephant image
#                         "type": "image_url",
#                         "image_url": {
#                             "url": "https://hatrabbits.com/wp-content/uploads/2017/01/random.jpg",
#                         },
#                     },
#                     {
#                         "type": "text",
#                         "text": "Describe the image",
#                     },
#                     {
#                         # Bird image
#                         "type": "image_url",
#                         "image_url": {
#                             "url": "data:image/png;base64,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"
#                         },
#                     },
#                 ],
#             },
#         ]
#     )

#     cassette_response = get_response_content_from_vcr(
#         vcr, adapter.get_model().get_path()
#     )

#     assert adapter_response.choices[0].message.content == cassette_response

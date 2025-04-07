from llm_adapters import OpenAI, AdapterFactory
from llm_adapters.types.model import ConversationRole
from llm_adapters.types.response import Conversation, Turn

client = OpenAI()

completion = client.chat.completions.create(
    model="claude-3-5-sonnet-20241022",
    messages=[
        {"role": "user", "content": "Hello!"},
    ],
)

print(completion.choices[0].message)

print("====== Adapter Factory ======")

adapter = AdapterFactory.get_adapter_by_path("claude-3-5-sonnet-20241022")

response = adapter.execute_sync(
    Conversation(
        [Turn(role=ConversationRole.user, content="How many toes does a dog have?")]
    )
)

print(response.choices[0].message)

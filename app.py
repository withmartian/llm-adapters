# import asyncio
# from llm_adapters import AsyncOpenAI


# async def main() -> None:
#     client = AsyncOpenAI()
#     completion = await client.chat.completions.create(
#         model="claude-3-5-sonnet-20241022",
#         messages=[{"role": "user", "content": "You are a helpful assistant."}],
#     )
#     print(completion)


# if __name__ == "__main__":
#     asyncio.run(main())

from llm_adapters import OpenAI

client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-3.5-turbo-instruct", prompt="You are a helpful assistant."
)

print(completion)

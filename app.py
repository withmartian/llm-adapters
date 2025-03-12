from openai import OpenAI
client = OpenAI()

batch_input_file = client.files.create(
    file=open("batchinput.jsonl", "rb"),
    purpose="batch"
)

print(batch_input_file)


# LLM Adapter

# import os
# from openai import llm_adapters

# client = OpenAI(
#     api_key=os.environ.get("OPENAI_API_KEY"),  # This is the default and can be omitted
# )

# chat_completion = client.chat.completions.create(
#     messages=[
#         {
#             "role": "user",
#             "content": "Say this is a test",
#         }
#     ],
#     model="gpt-4o",
# )

# print(chat_completion)
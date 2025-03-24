from llm_adapters import OpenAI

client = OpenAI()

# batch_input_file = client.files.create(
#     file=open("batchinput.jsonl", "rb"), purpose="batch"
# )

# print(batch_input_file)


# batch_input_file_id = batch_input_file.id
# batch_id = client.batches.create(
#     input_file_id=batch_input_file_id,
#     endpoint="/v1/chat/completions",
#     completion_window="24h",
#     metadata={"description": "nightly eval job"},
# )

# print("########################")

# print(batch_id)

# print("########################")

# id = batch_id.id

batch = client.batches.retrieve("batch_67e0fbff6c34819080be8a3567d141ea")

file_response = client.files.content(batch.output_file_id)
print(file_response.text)


print(client.batches.list(limit=10))

from adapters import AdapterFactory

# First component in model path is Provider, then Vendor, and last model name itself
adapter = AdapterFactory.get_adapter_by_path("openai/openai/gpt-4o-mini")

result = adapter.execute_sync(
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Write a haiku about recursion in programming.",
        },
    ]
)

# format result as a json
print(result)

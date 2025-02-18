# LLM Adapters

LLM Adapters provides a unified interface for interacting with various language model APIs, supporting both synchronous and asynchronous operations. Whether you're using OpenAI, Claude, or other providers, this library simplifies integration and offers flexibility in usage.

## Installation

To install llm-adapters for use in your projects, simply run:

```bash
pip install llm-adapters
```

## Usage

Package is using OpenAI SDK format, so you can use it as a drop in replacement for the OpenAI SDK, just need to change the import. It works both for synchronous and asynchronous clients.

```python
from llm_adapters import OpenAI

client = OpenAI()

completion = client.chat.completions.create(
  model="claude-3-5-sonnet-20241022",
  messages=[
    {"role": "developer", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ]
)

print(completion.choices[0].message)

```

Note: If you plan to contribute to this project, you'll need Poetry to set up the development environment. After cloning the repository, run:

```bash
poetry install
poetry run pre-commit install
```

## Getting Started

### Prerequisites

- Python 3.11+
- [Poetry](https://python-poetry.org/docs/#installation) (if using Poetry)

### Setup

1. Copy `.env-example` to `.env` and fill in the necessary environment variables.

## Supported Models

Adapter paths follow the format `provider/vendor/model_name`. Use `AdapterFactory.get_supported_models()` to retrieve all supported models. For any model, calling `.get_path()` will return its adapter path.

## Configuration and Environment

LLM Adapters can be customized via environment variables. For example, configure the HTTP client settings with:

```env
ADAPTERS_MAX_KEEPALIVE_CONNECTIONS_PER_PROCESS = 100
ADAPTERS_MAX_CONNECTIONS_PER_PROCESS = 1000
ADAPTERS_HTTP_CONNECT_TIMEOUT = 5
ADAPTERS_HTTP_TIMEOUT = 600
```

To override base URLs for testing or specific setups:

```env
_ADAPTERS_OVERRIDE_ALL_BASE_URLS_ = "https://new-base-url.com/api"
```

## Testing and Pre-commit

- Run tests with:

```bash
poetry run pytest
```

- Validate code using pre-commit hooks:

```bash
poetry run pre-commit run --all-files
```

## Contributing

We welcome contributions! To contribute:

1. Fork the repository.
2. Create a branch for your feature or bug fix.
3. Update tests and documentation as necessary.
4. Run existing tests and pre-commit checks.
5. Submit a pull request.

## Versioning

This project adheres to [Semantic Versioning](https://semver.org).

## Additional Resources

For complete documentation and a list of supported models, please visit our [Documentation Site](https://withmartian.github.io/llm-adapters/).

Happy coding!

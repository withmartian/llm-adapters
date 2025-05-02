import os

from llm_adapters.abstract_adapters import Adapter
from llm_adapters.abstract_adapters.provider_adapter_mixin import ProviderAdapterMixin

from llm_adapters.provider_adapters import (
    AnthropicSDKChatProviderAdapter,
    GeminiSDKChatProviderAdapter,
    AI21SDKChatProviderAdapter,
    LambdaLabsSDKChatProviderAdapter,
    OpenAISDKChatProviderAdapter,
    CerebrasSDKChatProviderAdapter,
    CohereSDKChatProviderAdapter,
    DeepInfraSDKChatProviderAdapter,
    TogetherSDKChatProviderAdapter,
    DeepSeekSDKChatProviderAdapter,
    AzureSDKChatProviderAdapter,
    GroqSDKChatProviderAdapter,
    PerplexitySDKChatProviderAdapter,
)
from llm_adapters.types import Model, AdapterModelNotFoundException

ADAPTER_CLASSES: list[type[Adapter]] = [
    AnthropicSDKChatProviderAdapter,
    GeminiSDKChatProviderAdapter,
    AI21SDKChatProviderAdapter,
    LambdaLabsSDKChatProviderAdapter,
    OpenAISDKChatProviderAdapter,
    TogetherSDKChatProviderAdapter,
    DeepSeekSDKChatProviderAdapter,
    CerebrasSDKChatProviderAdapter,
    CohereSDKChatProviderAdapter,
    DeepInfraSDKChatProviderAdapter,
    AzureSDKChatProviderAdapter,
    GroqSDKChatProviderAdapter,
    PerplexitySDKChatProviderAdapter,
]


class AdapterFactory:
    @staticmethod
    def _create_adapter_registry() -> dict[str, type[Adapter]]:
        adapters_classes: dict[str, type[Adapter]] = {}

        for adapter_class in ADAPTER_CLASSES:
            if issubclass(adapter_class, ProviderAdapterMixin):
                for model in adapter_class.get_supported_models():
                    adapters_classes[model.get_path()] = adapter_class
                    adapters_classes[model.name] = adapter_class

        return adapters_classes

    @staticmethod
    def _create_model_registry() -> dict[str, Model]:
        models: dict[str, Model] = {}

        for adapter_class in ADAPTER_CLASSES:
            if issubclass(adapter_class, ProviderAdapterMixin):
                for model in adapter_class.get_supported_models():
                    models[model.get_path()] = model
                    models[model.name] = model

        return models

    @staticmethod
    def _create_model_list() -> list[Model]:
        models: list[Model] = []

        for adapter_class in ADAPTER_CLASSES:
            if issubclass(adapter_class, ProviderAdapterMixin):
                for model in adapter_class.get_supported_models():
                    models.append(model)

        return models

    _adapter_registry = _create_adapter_registry()

    _model_registry = _create_model_registry()
    _model_list = _create_model_list()

    @staticmethod
    def get_adapter_by_path(model_path: str) -> Adapter:
        adapter_class = AdapterFactory._adapter_registry.get(model_path)
        model = AdapterFactory._model_registry.get(model_path)

        if adapter_class is None or model is None:
            raise AdapterModelNotFoundException()

        adapter = adapter_class()

        if isinstance(adapter, ProviderAdapterMixin):
            adapter._set_current_model(model)

        return adapter

    @staticmethod
    def get_adapter(model: Model) -> Adapter:
        adapter_class = AdapterFactory._adapter_registry.get(model.get_path())

        if adapter_class is None:
            raise AdapterModelNotFoundException()

        adapter = adapter_class()

        if isinstance(adapter, ProviderAdapterMixin):
            adapter._set_current_model(model)

        return adapter

    @staticmethod
    def get_model_by_path(model_path: str) -> Model:
        model = AdapterFactory._model_registry.get(model_path)

        if model is None:
            raise AdapterModelNotFoundException()

        return model

    # TODO: add more supports filtering
    @staticmethod
    def get_supported_models(
        supports_streaming: bool = False,
        supports_vision: bool = False,
        supports_functions: bool = False,
        supports_tools: bool = False,
        supports_n: bool = False,
        supports_json_output: bool = False,
        supports_json_content: bool = False,
    ) -> list[Model]:
        disabled_models = os.getenv("ADAPTER_DISABLED_MODELS", "").split(",")

        return [
            model
            for model in AdapterFactory._model_list
            if (not supports_streaming or model.supports_streaming)
            and (not supports_vision or model.supports_vision)
            and (not supports_functions or model.supports_functions)
            and (not supports_tools or model.supports_tools)
            and (not supports_n or model.supports_n)
            and (not supports_json_output or model.supports_json_output)
            and (not supports_json_content or model.supports_json_content)
            and model.name not in disabled_models
        ]


__all__ = ["AdapterFactory"]

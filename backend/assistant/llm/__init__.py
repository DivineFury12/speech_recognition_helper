from ..config import Settings
from .base import LLMError, LLMProvider


def create_provider(settings: Settings) -> LLMProvider | None:
    provider = settings.resolved_llm_provider
    if provider == "rules":
        return None
    if provider == "gigachat":
        from .gigachat_provider import GigaChatProvider

        return GigaChatProvider(
            credentials=settings.gigachat_credentials,
            scope=settings.gigachat_scope,
            model=settings.gigachat_model,
            verify_ssl=settings.gigachat_verify_ssl,
            ca_bundle=settings.gigachat_ca_bundle,
            timeout=settings.gigachat_timeout,
        )
    if provider == "local":
        from .local_provider import LocalProvider

        return LocalProvider()
    raise LLMError(f"Неизвестный LLM_PROVIDER: {provider} (ожидается gigachat, local или rules)")


__all__ = ["LLMError", "LLMProvider", "create_provider"]

import os
from dataclasses import dataclass, field


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "").strip().lower())
    db_path: str = field(default_factory=lambda: os.getenv("DB_PATH", "data/assistant.db"))
    asr_enabled: bool = field(default_factory=lambda: _bool("ASR_ENABLED", True))
    cors_origins: list[str] = field(
        default_factory=lambda: [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
    )

    gigachat_credentials: str = field(default_factory=lambda: os.getenv("GIGACHAT_CREDENTIALS", ""))
    gigachat_scope: str = field(default_factory=lambda: os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"))
    gigachat_model: str = field(default_factory=lambda: os.getenv("GIGACHAT_MODEL", "GigaChat"))
    gigachat_verify_ssl: bool = field(default_factory=lambda: _bool("GIGACHAT_VERIFY_SSL", True))
    gigachat_ca_bundle: str = field(default_factory=lambda: os.getenv("GIGACHAT_CA_BUNDLE", ""))
    gigachat_timeout: float = field(default_factory=lambda: float(os.getenv("GIGACHAT_TIMEOUT", "30")))

    @property
    def resolved_llm_provider(self) -> str:
        if self.llm_provider:
            return self.llm_provider
        return "gigachat" if self.gigachat_credentials else "rules"


def get_settings() -> Settings:
    return Settings()

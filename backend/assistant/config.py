import os
from dataclasses import dataclass, field
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
BUNDLED_CA = BACKEND_DIR / "certs" / "russian_trusted_ca.pem"


def load_env_file(path: Path = BACKEND_DIR / ".env") -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        os.environ.setdefault(key, value.strip().strip('"').strip("'"))


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _credentials() -> str:
    return os.getenv("GIGACHAT_CREDENTIALS") or os.getenv("GIGAKEY", "")


def _ca_bundle() -> str:
    value = os.getenv("GIGACHAT_CA_BUNDLE", "")
    if value:
        return value
    return str(BUNDLED_CA) if BUNDLED_CA.is_file() else ""


@dataclass(frozen=True)
class Settings:
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "").strip().lower())
    db_path: str = field(default_factory=lambda: os.getenv("DB_PATH", "data/assistant.db"))
    asr_enabled: bool = field(default_factory=lambda: _bool("ASR_ENABLED", True))
    cors_origins: list[str] = field(
        default_factory=lambda: [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
    )

    gigachat_credentials: str = field(default_factory=_credentials)
    gigachat_scope: str = field(default_factory=lambda: os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"))
    gigachat_model: str = field(default_factory=lambda: os.getenv("GIGACHAT_MODEL", "GigaChat-2"))
    gigachat_verify_ssl: bool = field(default_factory=lambda: _bool("GIGACHAT_VERIFY_SSL", True))
    gigachat_ca_bundle: str = field(default_factory=_ca_bundle)
    gigachat_timeout: float = field(default_factory=lambda: float(os.getenv("GIGACHAT_TIMEOUT", "30")))

    access_code: str = field(default_factory=lambda: os.getenv("ACCESS_CODE", ""))
    session_secret: str = field(default_factory=lambda: os.getenv("SESSION_SECRET", ""))
    cookie_secure: bool = field(default_factory=lambda: _bool("COOKIE_SECURE", False))
    rate_limit_per_minute: int = field(default_factory=lambda: int(os.getenv("RATE_LIMIT_PER_MINUTE", "30")))
    frontend_dist: str = field(
        default_factory=lambda: os.getenv("FRONTEND_DIST", str(BACKEND_DIR.parent / "frontend" / "dist"))
    )

    @property
    def resolved_llm_provider(self) -> str:
        if self.llm_provider:
            return self.llm_provider
        return "gigachat" if self.gigachat_credentials else "rules"


def get_settings() -> Settings:
    load_env_file()
    return Settings()

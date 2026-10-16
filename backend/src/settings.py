import dataclasses
import os
import pathlib

from sqlalchemy.engine import URL

from src.db_config import get_database_url

BACKEND_DIR = pathlib.Path(__file__).resolve().parents[1]


def load_env_file(path: pathlib.Path = BACKEND_DIR / '.env') -> None:
    """Read backend/.env for local runs; real environment variables win."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, _, value = line.partition('=')
        key = key.strip().removeprefix('export ').strip()
        os.environ.setdefault(key, value.strip().strip('"').strip("'"))


def _bundled_ca() -> str | None:
    """Local runs outside Docker: the Russian root and intermediate certificates shipped in certs/."""
    bundle = BACKEND_DIR / 'certs' / 'russian_trusted_ca.pem'
    return str(bundle) if bundle.is_file() else None


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name, '').strip().lower()
    return default if not value else value in {'1', 'true', 'yes', 'on'}


@dataclasses.dataclass(frozen=True)
class Settings:
    model_name: str
    device: str
    cache_dir: pathlib.Path
    ffmpeg_timeout_seconds: int
    gigachat_auth_key: str | None = dataclasses.field(repr=False)
    gigachat_scope: str
    gigachat_model: str
    gigachat_ca_bundle: str | None
    database_url: URL | None = dataclasses.field(default=None, repr=False)
    embedding_model_name: str = 'intfloat/multilingual-e5-small'
    voice_enabled: bool = True
    tts_enabled: bool = True
    tts_model_dir: pathlib.Path = BACKEND_DIR / 'cache' / 'tts'
    data_dir: pathlib.Path = BACKEND_DIR / 'data'
    access_code: str = dataclasses.field(default='', repr=False)
    session_secret: str = dataclasses.field(default='', repr=False)
    cookie_secure: bool = False
    rate_limit_per_minute: int = 30
    cors_origins: tuple[str, ...] = ()
    frontend_dist: pathlib.Path | None = None

    @classmethod
    def from_env(cls) -> 'Settings':
        return cls(
            model_name=os.getenv(
                'GIGAAM_MODEL', 'v3_e2e_rnnt'
            ),
            device=os.getenv(
                'GIGAAM_DEVICE', 'auto'
            ).lower(),
            cache_dir=pathlib.Path(
                os.getenv('GIGAAM_CACHE_DIR', '/cache/gigaam')
            ),
            ffmpeg_timeout_seconds=int(
                os.getenv('FFMPEG_TIMEOUT_SECONDS', '120')
            ),
            gigachat_auth_key=(os.getenv('GIGACHAT_AUTH_KEY') or os.getenv('GIGACHAT_CREDENTIALS')
                               or os.getenv('GIGAKEY') or '').strip().strip('"').strip("'") or None,
            gigachat_scope=os.getenv('GIGACHAT_SCOPE', 'GIGACHAT_API_PERS'),
            gigachat_model=os.getenv('GIGACHAT_MODEL', 'GigaChat-2'),
            gigachat_ca_bundle=os.getenv('GIGACHAT_CA_BUNDLE', '').strip() or _bundled_ca(),
            database_url=get_database_url(),
            embedding_model_name=os.getenv('EMBEDDING_MODEL', 'intfloat/multilingual-e5-small'),
            voice_enabled=_bool('VOICE_ENABLED', True),
            tts_enabled=_bool('VOICE_ENABLED', True) and _bool('TTS_ENABLED', True),
            tts_model_dir=pathlib.Path(os.getenv('TTS_MODEL_DIR', str(BACKEND_DIR / 'cache' / 'tts'))),
            data_dir=pathlib.Path(os.getenv('DATA_DIR', str(BACKEND_DIR / 'data'))),
            access_code=os.getenv('ACCESS_CODE', '').strip(),
            session_secret=os.getenv('SESSION_SECRET', '').strip(),
            cookie_secure=_bool('COOKIE_SECURE', False),
            rate_limit_per_minute=int(os.getenv('RATE_LIMIT_PER_MINUTE', '30')),
            cors_origins=tuple(o.strip() for o in os.getenv('CORS_ORIGINS', '').split(',') if o.strip()),
            frontend_dist=pathlib.Path(os.getenv('FRONTEND_DIST')) if os.getenv('FRONTEND_DIST') else None,
        )

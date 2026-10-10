import dataclasses
import os
import pathlib


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
            gigachat_auth_key=os.getenv('GIGACHAT_AUTH_KEY', '').strip() or None,
            gigachat_scope=os.getenv('GIGACHAT_SCOPE', 'GIGACHAT_API_PERS'),
            gigachat_model=os.getenv('GIGACHAT_MODEL', 'GigaChat-2'),
            gigachat_ca_bundle=os.getenv('GIGACHAT_CA_BUNDLE', '').strip() or None,
        )

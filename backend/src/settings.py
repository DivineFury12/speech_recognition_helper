import dataclasses
import os
import pathlib


@dataclasses.dataclass(frozen=True)
class Settings:
    model_name: str
    device: str
    cache_dir: pathlib.Path
    ffmpeg_timeout_seconds: int

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
        )
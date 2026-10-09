import pathlib
import subprocess

import src.errors


class FFmpegConverter:
    
    def __init__(self, timeout_seconds: int = 120):
        self.timeout_seconds = timeout_seconds
        
    
    def convert(
        self,
        source: pathlib.Path,
        destination: pathlib.Path,
    ) -> None:
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            str(destination),
        ]
        
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise src.errors.AudioConversionError(
                "Превышено время преобразования аудио"
            ) from exc
        except FileNotFoundError as exc:
            raise src.errors.AudioConversionError(
                "FFmpeg не установлен или недоступен"
            ) from exc

        if result.returncode != 0:
            # stderr можно записать в лог, не отдавая клиенту
            # подробности работы FFmpeg.
            raise src.errors.AudioConversionError(
                "Не удалось декодировать аудиофайл"
            )
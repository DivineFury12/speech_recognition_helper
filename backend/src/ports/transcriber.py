import pathlib
import typing

import src.schemas.transcribe


class Transcriber(typing.Protocol):
    @property
    def device(self) -> str: ...
    
    def transcribe(self, audio_path: pathlib.Path) -> str: ...
    
    def transcribe_longform(
        self,
        audio_path: pathlib.Path
    ) -> list[src.schemas.transcribe.AudioSegment]:
        ...


class AudioConverter(typing.Protocol):
    
    def convert(
        self,
        source: pathlib.Path,
        destination: pathlib.Path,
    ) -> None:
        ...

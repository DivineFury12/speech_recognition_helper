import pathlib
import tempfile
import typing
import shutil

import src.ports.transcriber
import src.schemas.transcribe

T = typing.TypeVar('T')


class TranscriptionService:
    
    def __init__(
        self,
        converter: src.ports.transcriber.AudioConverter,
        transcriber: src.ports.transcriber.Transcriber,
        ):
        self._converter = converter
        self._transcriber = transcriber
    
    def _run(
        self,
        source: typing.BinaryIO,
        filename: str | None,
        inference: typing.Callable[[pathlib.Path], T],
    ) -> T:
        suffix: str = pathlib.Path(filename or '').suffix.lower()
        
        if (
            not suffix
            or len(suffix) > 12
            or not suffix[1:].isalnum()
        ):
            suffix = ".bin"
        
        with tempfile.TemporaryDirectory(prefix='gigaam-') as tmp:
            directory = pathlib.Path(tmp)
            source_path = directory / f'input{suffix}'
            wav_path = directory / 'audio.wav'

            if source.seekable():
                source.seek(0)
            
            with source_path.open('wb') as output:
                shutil.copyfileobj(
                    source,
                    output,
                    length=1024 * 1024,
                )
            
            self._converter.convert(source_path, wav_path)
            
            return inference(wav_path)
    
    def transribe(
        self,
        source: typing.BinaryIO,
        filename: str | None = None,
    ) -> str:
        return self._run(
            source,
            filename,
            self._transcriber.transcribe
        )
    
    def transcribe_longform(
        self,
        source: typing.BinaryIO,
        filename: str | None = None,
    ) -> list[src.schemas.transcribe.TranscriptSegment]:
        return self._run(
            source,
            filename,
            self._transcriber.transcribe_longform,
        )
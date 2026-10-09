import pathlib
import threading

import gigaam
import torch

import src.errors
import src.schemas.transcribe

class GigaAMTransciber:
    
    def __init__(self, model, device: str):
        self._model = model
        self._device = device
        self._infer_lock = threading.Lock()
    
    @classmethod
    def load(
        cls,
        model_name: str,
        requested_device: str,
        cache_dir: pathlib.Path, 
    ) -> 'GigaAMTransciber':
        if requested_device not in {'auto', 'cpu', 'cuda'}:
            raise ValueError('GIGAAM_DEVICE должен быть auto, cpu или cuda')
        
        if requested_device == 'auto':
            device = (
                'cuda' if torch.cuda.is_available() else 'cpu'
            )
        
        else:
            device = requested_device
        
        if device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "GPU запрошен, но CUDA недоступна"
                )

            torch.ones(1, device="cuda")
            torch.cuda.synchronize()
        
        model = gigaam.load_model(
            model_name,
            device=device,
            download_root=str(cache_dir)
        )
        
        actual_device = str(next(model.parameters()).device)
        return cls(model, actual_device)
    
    @property
    def device(self) -> str:
        return self._device

    def transcribe(self, audio_path: pathlib.Path) -> str:
        try:
            with self._infer_lock:
                result = self._model.transcribe(str(audio_path))
            return result.text
        except ValueError as exc:
            raise src.errors.InvalidAudioError(
                "Модель не смогла обработать аудиофайл"
            ) from exc
        except Exception as exc:
            raise src.errors.TranscriptionError(
                "Ошибка при распознавании аудио"
            ) from exc
    
    def transcribe_longform(
        self,
        audio_path: pathlib.Path,
    ) -> list[src.schemas.transcribe.TranscriptSegment]:
        try:
            with self._infer_lock:
                result = self._model.transcribe_longform(
                    str(audio_path),
                    fr_batch_size=1,
                )
            return [src.schemas.transcribe.TranscriptSegment(
                start=segment.start,
                end=segment.end,
                text=segment.text,
            )
            for segment in result.segments
            ]
        
        except Exception as exc:
            raise src.errors.TranscriptionError(
                "Ошибка longform-распознавания"
            ) from exc

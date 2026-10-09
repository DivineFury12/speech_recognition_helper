class AudioConversionError(Exception):
    """Не удалось преобразовать аудиофайл."""


class InvalidAudioError(Exception):
    """Аудиофайл не удалось обработать как входные данные."""


class TranscriptionError(Exception):
    """Ошибка выполнения распознавания."""
import pydantic


class TranscribeResponse(pydantic.BaseModel):
    text: str


class TranscriptSegment(pydantic.BaseModel):
    start: float
    end: float
    text: str


class TranscribeLongformResponse(pydantic.BaseModel):
    segments: list[TranscriptSegment]
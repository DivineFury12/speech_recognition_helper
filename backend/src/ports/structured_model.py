import typing


class StructuredModelClient(typing.Protocol):
    """Provider adapter that makes an HTTP call and returns one JSON object."""

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict,
    ) -> dict:
        ...

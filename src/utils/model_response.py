from typing import Any

from pydantic import BaseModel


ERROR_MAP: dict[int, str] = {
    429: "rate_limit",
    599: "connection_error",
    401: "api_error",
    408: "request_timeout",
    500: "internal_server",
}


class Performance(BaseModel):
    """Python object for wrapping performance info from a model."""

    latency: float
    prompt_tokens: float
    response_tokens: float
    reasoning_tokens: float | None
    time_per_token: float | None
    relative_output_tokens: float


class ErrorTracker(BaseModel):
    """Python object for tracking model errors.

    Codes currently tracking:
    - 429: rate limit
    - 599: connection error
    - 401: API error
    - 408: request timeout
    - 500: internal server
    - Other: Any other errors encountered.
    """

    rate_limit: int = 0
    connection_error: int = 0
    api_error: int = 0
    request_timeout: int = 0
    internal_server: int = 0
    other: int = 0

    def increment(self, response_code: int):
        """Increment the appropriate error counter based on the response code."""
        error_key = ERROR_MAP.get(response_code, "other")
        setattr(self, error_key, getattr(self, error_key) + 1)


class ModelResponse(BaseModel):
    """Python object for wrapping a model response from LLM."""

    input_prompt: str | list
    llm_response: str
    raw_response: str
    raw_response_object: Any | None = None
    is_failure: bool | None = None
    reasoning_response: str | None = None
    reasoning_possible_to_extract: bool | None = None
    reasoning_correct_format: bool | None = None
    response_code: int
    stop_reason: str | None = None
    tool_calls: list | None = None
    performance: Performance | None
    wait_time: int | None = None
    error_tracker: ErrorTracker | None = None
    model_parameters: dict[str, Any] | list
    model_info: dict[str, Any] | None = None
    agent_messages: list | None = None

    def __init__(self, **data):
        super().__init__(**data)
        self.input_prompt = self.input_prompt
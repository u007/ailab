from typing import Any, Literal

from pydantic import BaseModel, Field


class ImageUrl(BaseModel):
    url: str
    detail: str | None = None


class ChatContentPart(BaseModel):
    type: Literal["text", "image_url", "input_text", "input_image"]
    text: str | None = None
    image_url: ImageUrl | str | None = None
    image: ImageUrl | str | None = None
    file_id: str | None = None


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[ChatContentPart] | None = None
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, ge=0, le=1)
    stream: bool = False
    stop: str | list[str] | None = None
    user: str | None = None

    # Accept common OpenAI-compatible pass-through fields without rejecting clients.
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    response_format: dict[str, Any] | None = None
    seed: int | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None


class ResponseInputMessage(BaseModel):
    role: Literal["system", "user", "assistant", "developer"] = "user"
    content: str | list[dict[str, Any]]


class ResponsesRequest(BaseModel):
    model: str
    input: str | list[ResponseInputMessage] | list[dict[str, Any]]
    instructions: str | None = None
    max_output_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, ge=0, le=1)
    stream: bool = False
    stop: str | list[str] | None = None
    user: str | None = None


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "nvidia"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]

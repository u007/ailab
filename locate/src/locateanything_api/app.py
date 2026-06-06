from __future__ import annotations

import json
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from locateanything_api import __version__
from locateanything_api.backend import LocateAnythingBackend
from locateanything_api.config import Settings, get_settings
from locateanything_api.schemas import ChatCompletionRequest, ChatMessage, ModelCard, ModelList, ResponsesRequest


def create_app(settings: Settings | None = None, backend: LocateAnythingBackend | None = None) -> FastAPI:
    app_settings = settings or get_settings()
    app_backend = backend or LocateAnythingBackend(app_settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if app_settings.preload_model:
            app_backend.load()
        yield

    app = FastAPI(
        title="LocateAnything OpenAI-Compatible API",
        version=__version__,
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        client = request.client.host if request.client else "-"
        start = time.perf_counter()
        print(f"--> {request.method} {request.url.path} from {client}", flush=True)
        try:
            response = await call_next(request)
        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            print(f"<-- {request.method} {request.url.path} 500 {duration_ms:.1f}ms ({exc})", flush=True)
            raise
        duration_ms = (time.perf_counter() - start) * 1000
        print(f"<-- {request.method} {request.url.path} {response.status_code} {duration_ms:.1f}ms", flush=True)
        return response

    def get_backend() -> LocateAnythingBackend:
        return app_backend

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {
            "status": "ok",
            "model": app_settings.model_id,
            "model_loaded": getattr(app_backend, "is_loaded", False),
            "model_loaded_at": getattr(app_backend, "loaded_at", None),
        }

    @app.get("/v1/models", response_model=ModelList)
    async def models() -> ModelList:
        return ModelList(data=[ModelCard(id=app_settings.model_id, created=int(time.time()))])

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(
        request: ChatCompletionRequest,
        current_backend: LocateAnythingBackend = Depends(get_backend),
    ) -> dict[str, object] | StreamingResponse:
        if request.model not in {app_settings.model_id, "locateanything"}:
            raise HTTPException(status_code=404, detail=f"Model {request.model!r} is not served by this API.")

        try:
            content = await current_backend.complete(request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            log_exception("/v1/chat/completions", exc)
            raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

        response_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        if request.stream:
            log_response_payload(
                app_settings,
                "/v1/chat/completions stream",
                {
                    "id": response_id,
                    "model": request.model,
                    "content": content,
                    "usage": chat_usage(request, content),
                },
            )
            return StreamingResponse(
                stream_chat_completion(response_id, created, request.model, content),
                media_type="text/event-stream",
            )

        payload = {
            "id": response_id,
            "object": "chat.completion",
            "created": created,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": chat_usage(request, content),
        }
        log_response_payload(app_settings, "/v1/chat/completions", payload)
        return payload

    @app.post("/v1/responses", response_model=None)
    async def responses(
        request: ResponsesRequest,
        current_backend: LocateAnythingBackend = Depends(get_backend),
    ) -> dict[str, object] | StreamingResponse:
        chat_request = responses_to_chat_request(request)
        if request.model not in {app_settings.model_id, "locateanything"}:
            raise HTTPException(status_code=404, detail=f"Model {request.model!r} is not served by this API.")

        try:
            content = await current_backend.complete(chat_request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            log_exception("/v1/responses", exc)
            raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

        response_id = f"resp_{uuid.uuid4().hex}"
        created = int(time.time())

        if request.stream:
            log_response_payload(
                app_settings,
                "/v1/responses stream",
                {
                    "id": response_id,
                    "model": request.model,
                    "output_text": content,
                    "usage": responses_usage(request, content),
                },
            )
            return StreamingResponse(
                stream_response(response_id, created, request.model, content),
                media_type="text/event-stream",
            )

        payload = {
            "id": response_id,
            "object": "response",
            "created_at": created,
            "status": "completed",
            "model": request.model,
            "output": [
                {
                    "id": f"msg_{uuid.uuid4().hex}",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": content,
                            "annotations": [],
                        }
                    ],
                }
            ],
            "output_text": content,
            "usage": responses_usage(request, content),
        }
        log_response_payload(app_settings, "/v1/responses", payload)
        return payload

    return app


def log_response_payload(settings: Settings, endpoint: str, payload: dict[str, object]) -> None:
    if settings.log_responses:
        body = json.dumps(payload, separators=(",", ":"))
        print(f"Sending {endpoint} response: {body}", flush=True)


def log_exception(endpoint: str, exc: Exception) -> None:
    print(f"Unhandled {endpoint} error: {exc}", flush=True)
    traceback.print_exc()


def chat_usage(request: ChatCompletionRequest, content: str) -> dict[str, int]:
    prompt_tokens = estimate_token_count(json.dumps([message.model_dump() for message in request.messages]))
    completion_tokens = estimate_token_count(content)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def responses_usage(request: ResponsesRequest, content: str) -> dict[str, int]:
    input_tokens = estimate_token_count(json.dumps(request.model_dump(mode="json")["input"]))
    output_tokens = estimate_token_count(content)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def estimate_token_count(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def responses_to_chat_request(request: ResponsesRequest) -> ChatCompletionRequest:
    messages: list[ChatMessage] = []

    if isinstance(request.input, str):
        messages.append(ChatMessage(role="user", content=request.input))
    else:
        for item in request.input:
            item_dict = item.model_dump() if hasattr(item, "model_dump") else dict(item)
            if item_dict.get("type") == "message" and "content" in item_dict:
                role = normalize_role(item_dict.get("role", "user"))
                messages.append(ChatMessage(role=role, content=normalize_response_content(item_dict["content"])))
            elif "role" in item_dict and "content" in item_dict:
                role = normalize_role(item_dict.get("role", "user"))
                messages.append(ChatMessage(role=role, content=normalize_response_content(item_dict["content"])))
            elif item_dict.get("type") in {"input_text", "input_image"}:
                messages.append(ChatMessage(role="user", content=normalize_response_content([item_dict])))

    return ChatCompletionRequest(
        model=request.model,
        messages=messages,
        max_tokens=request.max_output_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stream=request.stream,
        stop=request.stop,
        user=request.user,
    )


def normalize_role(role: object) -> str:
    if role in {"developer", "system"}:
        return "system"
    return str(role or "user")


def normalize_response_content(content: object) -> str | list[dict[str, object]]:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    normalized: list[dict[str, object]] = []
    for part in content:
        part_dict = part.model_dump() if hasattr(part, "model_dump") else dict(part)
        part_type = part_dict.get("type")
        if part_type == "input_text":
            normalized.append({"type": "input_text", "text": part_dict.get("text", "")})
        elif part_type == "output_text":
            normalized.append({"type": "text", "text": part_dict.get("text", "")})
        elif part_type == "input_image":
            image_url = part_dict.get("image_url") or part_dict.get("image")
            normalized.append({"type": "input_image", "image_url": image_url})
        elif part_type == "image_url":
            normalized.append(part_dict)
        elif part_type == "text":
            normalized.append(part_dict)
    return normalized


async def stream_chat_completion(response_id: str, created: int, model: str, content: str) -> AsyncIterator[str]:
    chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": content},
                "finish_reason": None,
            }
        ],
    }
    yield f"data: {json.dumps(chunk)}\n\n"

    done = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(done)}\n\n"
    yield "data: [DONE]\n\n"


async def stream_response(response_id: str, created: int, model: str, content: str) -> AsyncIterator[str]:
    created_event = {
        "type": "response.created",
        "response": {
            "id": response_id,
            "object": "response",
            "created_at": created,
            "status": "in_progress",
            "model": model,
        },
    }
    yield f"data: {json.dumps(created_event)}\n\n"

    text_event = {
        "type": "response.output_text.delta",
        "item_id": f"msg_{uuid.uuid4().hex}",
        "output_index": 0,
        "content_index": 0,
        "delta": content,
    }
    yield f"data: {json.dumps(text_event)}\n\n"

    completed_event = {
        "type": "response.completed",
        "response": {
            "id": response_id,
            "object": "response",
            "created_at": created,
            "status": "completed",
            "model": model,
            "output_text": content,
        },
    }
    yield f"data: {json.dumps(completed_event)}\n\n"
    yield "data: [DONE]\n\n"

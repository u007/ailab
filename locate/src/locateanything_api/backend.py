from __future__ import annotations

import base64
import binascii
import json
import re
import threading
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import httpx
from PIL import Image

from locateanything_api.config import Settings
from locateanything_api.schemas import ChatCompletionRequest, ChatMessage


@dataclass(frozen=True)
class PreparedPrompt:
    prompt: str
    images: list[Image.Image]


class LocateAnythingBackend:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._tokenizer: Any | None = None
        self._processor: Any | None = None
        self._model: Any | None = None
        self._load_lock = threading.Lock()
        self._generate_lock = threading.Lock()
        self._loaded_at: float | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def loaded_at(self) -> float | None:
        return self._loaded_at

    def load(self) -> None:
        if self._model is not None:
            return

        with self._load_lock:
            if self._model is not None:
                return

            import torch
            from transformers import AutoModel, AutoProcessor, AutoTokenizer

            dtype = self._resolve_dtype(torch)
            common_kwargs = {"trust_remote_code": self.settings.trust_remote_code}
            model_kwargs = dict(common_kwargs)
            model_kwargs["dtype"] = dtype
            if self.settings.device_map:
                model_kwargs["device_map"] = self.settings.device_map

            self._tokenizer = AutoTokenizer.from_pretrained(self.settings.model_id, **common_kwargs)
            self._processor = AutoProcessor.from_pretrained(self.settings.model_id, **common_kwargs)
            self._model = AutoModel.from_pretrained(self.settings.model_id, **model_kwargs)
            if not self.settings.device_map:
                self._model = self._model.to(self._resolve_device(torch))
            self._model.eval()
            self._loaded_at = time.time()

    async def complete(self, request: ChatCompletionRequest) -> str:
        self.load()
        prepared = await prepare_prompt(
            request.messages,
            self.settings.request_timeout_seconds,
            min_image_pixels=self.settings.min_image_pixels,
            max_image_pixels=self.settings.max_image_pixels,
            max_image_side=self.settings.max_image_side,
        )
        if self.settings.prompt_mode == "template":
            prepared = PreparedPrompt(prompt=normalize_locate_prompt(prepared.prompt), images=prepared.images)
        if self.settings.log_prompts:
            print(f"LocateAnything prompt: {prepared.prompt}", flush=True)

        assert self._model is not None
        assert self._tokenizer is not None
        assert self._processor is not None

        generation_kwargs = {
            "max_new_tokens": request.max_tokens or self.settings.max_new_tokens,
            "generation_mode": self.settings.generation_mode,
            "temperature": 0.7 if request.temperature is None else request.temperature,
            "do_sample": True,
            "top_p": request.top_p if request.top_p is not None else self.settings.top_p,
            "repetition_penalty": self.settings.repetition_penalty,
            "verbose": self.settings.verbose_generation,
        }

        with self._generate_lock:
            try:
                output = self._generate(prepared, generation_kwargs)
            finally:
                if self.settings.empty_cuda_cache_after_generate:
                    empty_cuda_cache()
        text = coerce_model_text(output, self._tokenizer)
        text = apply_stop_sequences(text, request.stop)
        if self.settings.log_responses:
            print(f"Raw model output: {text}", flush=True)
        return json.dumps(parse_bboxes(text))

    def _generate(self, prepared: PreparedPrompt, generation_kwargs: dict[str, Any]) -> Any:
        if not prepared.images:
            raise ValueError("LocateAnything requests require at least one image.")

        messages = [
            {
                "role": "user",
                "content": [
                    *({"type": "image", "image": image} for image in prepared.images),
                    {"type": "text", "text": prepared.prompt},
                ],
            }
        ]

        if hasattr(self._processor, "py_apply_chat_template"):
            text = self._processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        images, videos = self._processor.process_vision_info(messages)
        inputs = self._processor(text=[text], images=images, videos=videos, return_tensors="pt")
        inputs = move_inputs_to_model(inputs, self._model)

        model_dtype = infer_model_dtype(self._model)
        pixel_values = inputs["pixel_values"].to(model_dtype) if model_dtype is not None else inputs["pixel_values"]
        image_grid_hws = inputs.get("image_grid_hws", None)

        import torch

        with torch.inference_mode():
            return self._model.generate(
                pixel_values=pixel_values,
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_grid_hws=image_grid_hws,
                tokenizer=self._tokenizer,
                use_cache=True,
                **generation_kwargs,
            )

    def _resolve_dtype(self, torch: Any) -> Any:
        dtype = self.settings.torch_dtype
        if dtype == "auto":
            if torch.cuda.is_available():
                if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                    return torch.bfloat16
                return torch.float16
            return torch.float32
        if not hasattr(torch, dtype):
            raise ValueError(f"Unsupported LOCATE_TORCH_DTYPE={dtype!r}")
        return getattr(torch, dtype)

    def _resolve_device(self, torch: Any) -> str:
        if self.settings.device != "auto":
            return self.settings.device
        return "cuda" if torch.cuda.is_available() else "cpu"


async def prepare_prompt(
    messages: list[ChatMessage],
    timeout_seconds: float,
    min_image_pixels: int = 0,
    max_image_pixels: int = 0,
    max_image_side: int = 0,
) -> PreparedPrompt:
    prompt_parts: list[str] = []
    images: list[Image.Image] = []

    for message in messages:
        if message.content is None:
            continue
        if message.role not in {"user", "tool"}:
            continue
        if isinstance(message.content, str):
            text = clean_request_prompt_text(message.content)
            if text:
                prompt_parts.append(text)
            continue

        part_texts: list[str] = []
        for part in message.content:
            if part.type in {"text", "input_text"} and part.text:
                text = clean_request_prompt_text(part.text)
                if text:
                    part_texts.append(text)
            elif part.type in {"image_url", "input_image"}:
                image_ref = part.image_url or part.image
                if not image_ref:
                    continue
                url = image_ref.url if hasattr(image_ref, "url") else image_ref
                image = await load_image(str(url), timeout_seconds)
                images.append(resize_image_for_memory(image, min_image_pixels, max_image_pixels, max_image_side))

        if part_texts:
            prompt_parts.append(" ".join(part_texts))

    if not prompt_parts:
        raise ValueError("At least one text prompt is required.")

    return PreparedPrompt(prompt="\n".join(prompt_parts), images=images)


def resize_image_for_memory(
    image: Image.Image,
    min_pixels: int,
    max_pixels: int,
    max_side: int,
) -> Image.Image:
    width, height = image.size
    scale = 1.0

    if min_pixels and width * height < min_pixels:
        scale = max(scale, (min_pixels / (width * height)) ** 0.5)
    if max_pixels and width * height * scale * scale > max_pixels:
        scale = min(scale, (max_pixels / (width * height)) ** 0.5)
    if max_side and max(width, height) * scale > max_side:
        scale = min(scale, max_side / max(width, height))
    if scale == 1.0:
        return image

    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    if new_size == image.size:
        return image
    resized = image.resize(new_size, Image.Resampling.LANCZOS)
    return resized


def clean_request_prompt_text(text: str) -> str:
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        normalized = line.strip()
        if not normalized:
            continue
        if is_output_schema_line(normalized):
            continue
        cleaned_lines.append(normalized)
    return " ".join(cleaned_lines).strip()


def is_output_schema_line(line: str) -> bool:
    lowered = line.lower()
    schema_terms = (
        "bbox_2d",
        "bounding box",
        "json",
        "schema",
        "return format",
        "respond",
        "response",
        "label",
        "array",
        "coordinates",
        "y1",
        "x1",
        "y2",
        "x2",
    )
    return any(term in lowered for term in schema_terms)


def normalize_locate_prompt(prompt: str) -> str:
    prompt = re.sub(r"\s+", " ", prompt).strip()
    lowered = prompt.lower()

    if not prompt:
        return "Detect all the text in box format."
    if "all text" in lowered or "detect text" in lowered or "ocr" in lowered:
        return "Detect all the text in box format."
    if "text referred" in lowered:
        return prompt
    if lowered.startswith("point to"):
        return prompt
    if lowered.startswith("locate "):
        return prompt
    if lowered.startswith("find "):
        phrase = prompt[5:].strip(" .")
        return f"Locate all the instances that match the following description: {phrase}."
    if lowered.startswith("detect "):
        phrase = prompt[7:].strip(" .")
        return f"Locate all the instances that matches the following description: {phrase}."
    return f"Locate all the instances that match the following description: {prompt}."


async def load_image(url: str, timeout_seconds: float) -> Image.Image:
    if url.startswith("data:"):
        return decode_data_url(url)

    async with httpx.AsyncClient(timeout=timeout_seconds, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
        return image_from_bytes(response.content)


def decode_data_url(url: str) -> Image.Image:
    try:
        _, encoded = url.split(",", 1)
        return image_from_bytes(base64.b64decode(encoded, validate=True))
    except (ValueError, binascii.Error) as exc:
        raise ValueError("Invalid image data URL.") from exc


def image_from_bytes(data: bytes) -> Image.Image:
    image = Image.open(BytesIO(data))
    return image.convert("RGB")


def coerce_model_text(output: Any, tokenizer: Any) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        for key in ("text", "answer", "response", "content"):
            if key in output:
                return str(output[key])
    if isinstance(output, (list, tuple)) and output:
        if all(isinstance(item, int) for item in output):
            return tokenizer.decode(output, skip_special_tokens=True)
        return str(output[0])
    return str(output)


def parse_bboxes(answer: str) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []
    last_end = 0
    current_label = ""

    pattern = r"<box>\s*(?:<(\d+)><(\d+)><(\d+)><(\d+)>|(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+))\s*</box>"
    for match in re.finditer(pattern, answer):
        label = clean_label(answer[last_end : match.start()])
        if label:
            current_label = label

        groups = [group for group in match.groups() if group is not None]
        x1, y1, x2, y2 = [int(group) for group in groups]
        left, right = sorted((clamp_normalized_coord(x1), clamp_normalized_coord(x2)))
        top, bottom = sorted((clamp_normalized_coord(y1), clamp_normalized_coord(y2)))

        boxes.append(
            {
                "bbox_2d": [top, left, bottom, right],
                "label": current_label,
            }
        )
        last_end = match.end()

    return boxes


def clean_label(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \n\t\r:;,.-")


def clamp_normalized_coord(value: int) -> int:
    return max(0, min(1000, value))


def infer_model_dtype(model: Any) -> Any:
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return None


def apply_stop_sequences(text: str, stop: str | list[str] | None) -> str:
    if not stop:
        return text
    stops = [stop] if isinstance(stop, str) else stop
    indexes = [text.find(sequence) for sequence in stops if sequence and text.find(sequence) >= 0]
    return text[: min(indexes)] if indexes else text


def move_inputs_to_model(inputs: Any, model: Any) -> Any:
    try:
        device = next(model.parameters()).device
    except StopIteration:
        return inputs

    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return

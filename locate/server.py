"""Serve LocateAnything-3B model using FastAPI."""
import base64
from io import BytesIO
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor

app = FastAPI(title="LocateAnything-3B Server")

# Global model/processor (loaded once at startup)
model = None
tokenizer = None
processor = None

MODEL_NAME = "nvidia/LocateAnything-3B"


class ChatMessage(BaseModel):
    role: str
    content: list


class ChatRequest(BaseModel):
    model: str = MODEL_NAME
    messages: list[ChatMessage]
    max_tokens: int = 512


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    choices: list
    usage: dict


def load_model():
    global model, tokenizer, processor
    print(f"Loading {MODEL_NAME}...")
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )
    print("Model loaded!")


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "owned_by": "nvidia",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(request: ChatRequest):
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # Build messages for the processor
    images = []

    for msg in request.messages:
        if isinstance(msg.content, list):
            for part in msg.content:
                if part.get("type") == "image_url":
                    image_url = part.get("image_url", {}).get("url", "")
                    if image_url.startswith("http"):
                        import urllib.request
                        with urllib.request.urlopen(image_url) as resp:
                            img = Image.open(BytesIO(resp.read())).convert("RGB")
                            images.append(img)
                    elif image_url.startswith("data:"):
                        # data:image/...;base64,...
                        _, b64data = image_url.split(",", 1)
                        img = Image.open(BytesIO(base64.b64decode(b64data))).convert("RGB")
                        images.append(img)

    if not images:
        raise HTTPException(
            status_code=400,
            detail="This model requires at least one image in the request",
        )

    # Build the prompt
    text_parts = []
    for msg in request.messages:
        if isinstance(msg.content, str):
            text_parts.append(msg.content)
        elif isinstance(msg.content, list):
            for part in msg.content:
                if part.get("type") == "text":
                    text_parts.append(part.get("text", ""))

    prompt = " ".join(text_parts) if text_parts else "Describe this image"

    # Use the model's generate method
    inputs = processor(text=prompt, images=images, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs.to(model.device if hasattr(model, 'device') else 'cpu'),
            max_new_tokens=request.max_tokens,
        )

    # Decode the output
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {
        "id": "chatcmpl-locate",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": generated},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    load_model()
    uvicorn.run(app, host="0.0.0.0", port=8001)

"""FastAPI server for audio transcription using parakeet-mlx."""

import os
import tempfile
import zipfile
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import aiofiles
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from parakeet_mlx import from_pretrained


class AudioFormat(str, Enum):
    """Supported audio formats for output."""

    WAV = "wav"
    TXT = "txt"
    JSON = "json"


class TranscriptionResult(BaseModel):
    """Transcription result with timestamps."""

    text: str
    sentences: list[dict]
    duration: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model: str


# Global model instance
_model = None


def get_model():
    """Get or load the parakeet model (lazy loading)."""
    global _model
    if _model is None:
        _model = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")
    return _model


app = FastAPI(
    title="Parakeet Transcription API",
    description="Audio transcription API using NVIDIA Parakeet TDT model (Apple Silicon optimized)",
    version="0.1.0",
)

# CORS middleware for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="ok", model="mlx-community/parakeet-tdt-0.6b-v3")


@app.post("/transcribe", response_model=TranscriptionResult)
async def transcribe_audio(
    file: Annotated[UploadFile, File(description="Audio file to transcribe")],
    output_format: Annotated[
        AudioFormat, Form(description="Output format for results")
    ] = AudioFormat.JSON,
):
    """
    Transcribe an audio file.

    Supports: wav, mp3, ogg, flac, m4a, and other common audio formats.
    """
    # Validate file type
    allowed_extensions = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac", ".wma", ".opus"}
    file_ext = Path(file.filename or "").suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{file_ext}'. Allowed: {', '.join(allowed_extensions)}",
        )

    # Save uploaded file to temp
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
    try:
        content = await file.read()
        temp_input.write(content)
        temp_input.close()

        # Transcribe
        model = get_model()
        result = model.transcribe(temp_input.name)

        # Build response
        sentences = [
            {
                "start": sentence.start,
                "end": sentence.end,
                "text": sentence.text,
                "confidence": sentence.confidence,
            }
            for sentence in result.sentences
        ]

        full_text = " ".join(s["text"] for s in sentences)
        duration = sentences[-1]["end"] if sentences else 0.0

        transcription = TranscriptionResult(
            text=full_text, sentences=sentences, duration=duration
        )

        return transcription

    finally:
        # Cleanup
        os.unlink(temp_input.name)


@app.post("/transcribe/file")
async def transcribe_to_file(
    file: Annotated[UploadFile, File(description="Audio file to transcribe")],
    format: Annotated[AudioFormat, Form(description="Output format")] = AudioFormat.TXT,
):
    """
    Transcribe audio and return results as a downloadable file.

    Returns a text file with timestamps or JSON format.
    """
    # Validate file type
    allowed_extensions = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac", ".wma", ".opus"}
    file_ext = Path(file.filename or "").suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{file_ext}'. Allowed: {', '.join(allowed_extensions)}",
        )

    # Create temp files
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=f".{format.value}")

    try:
        # Save uploaded file
        content = await file.read()
        temp_input.write(content)
        temp_input.close()

        # Transcribe
        model = get_model()
        result = model.transcribe(temp_input.name)

        # Write output based on format
        if format == AudioFormat.TXT:
            for sentence in result.sentences:
                line = f"[{sentence.start:.1f}s - {sentence.end:.1f}s] ({sentence.confidence:.0%}) {sentence.text}\n"
                temp_output.write(line.encode())
            media_type = "text/plain"
        elif format == AudioFormat.JSON:
            import json

            sentences = [
                {
                    "start": sentence.start,
                    "end": sentence.end,
                    "text": sentence.text,
                    "confidence": sentence.confidence,
                }
                for sentence in result.sentences
            ]
            temp_output.write(
                json.dumps(
                    {"text": " ".join(s["text"] for s in sentences), "sentences": sentences},
                    indent=2,
                ).encode()
            )
            media_type = "application/json"
        else:  # WAV - just return a text file for now
            for sentence in result.sentences:
                line = f"[{sentence.start:.1f}s - {sentence.end:.1f}s] ({sentence.confidence:.0%}) {sentence.text}\n"
                temp_output.write(line.encode())
            media_type = "text/plain"

        temp_output.close()

        # Return file
        filename = Path(file.filename or "audio").stem + f"_transcript.{format.value}"
        return FileResponse(
            temp_output.name,
            media_type=media_type,
            filename=filename,
            background=lambda: os.unlink(temp_output.name),
        )

    finally:
        os.unlink(temp_input.name)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

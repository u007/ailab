#!/usr/bin/env python3
"""
NVIDIA Nemotron Streaming ASR Sample

This script demonstrates how to use the nvidia/nemotron-speech-streaming-en-0.6b
model from Hugging Face for speech-to-text transcription.

Model: https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b

Requirements:
    apt-get update && apt-get install -y libsndfile1 ffmpeg
    pip install Cython packaging
    pip install git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]

The model accepts:
- Single-channel (mono) audio sampled at 16,000 Hz
- At least 80ms duration

Chunk sizes (att_context_size = [left, right]):
- [70, 0]: 80ms chunk (lowest latency, WER ~8.5%)
- [70, 1]: 160ms chunk (WER ~7.8%)
- [70, 6]: 560ms chunk (WER ~7.2%)
- [70, 13]: 1120ms chunk (best accuracy, WER ~7.2%)
"""

import os
import argparse
import wave
import numpy as np
from typing import Optional

try:
    import nemo.collections.asr as nemo_asr
    from omegaconf import OmegaConf
except ImportError:
    print("NeMo toolkit not found. Install with:")
    print("  pip install Cython packaging")
    print("  pip install git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]")
    exit(1)


def check_audio_file(audio_path: str) -> tuple[bool, str, Optional[dict]]:
    """
    Validate audio file meets model requirements.

    Returns:
        (is_valid, error_message, audio_info)
    """
    if not os.path.exists(audio_path):
        return False, f"Audio file not found: {audio_path}", None

    try:
        with wave.open(audio_path, 'rb') as wf:
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            frames = wf.getnframes()
            duration = frames / sample_rate

            audio_info = {
                'channels': channels,
                'sample_rate': sample_rate,
                'frames': frames,
                'duration': duration,
            }

            if channels != 1:
                return False, f"Audio must be mono (single channel). Got {channels} channels.", audio_info

            if sample_rate != 16000:
                return False, f"Audio must be 16kHz sample rate. Got {sample_rate}Hz.", audio_info

            if duration < 0.08:
                return False, f"Audio must be at least 80ms duration. Got {duration:.3f}s.", audio_info

            return True, "Valid", audio_info

    except Exception as e:
        return False, f"Failed to read audio file: {e}", None


def convert_to_16khz_mono(input_path: str, output_path: str) -> bool:
    """
    Convert audio file to 16kHz mono using ffmpeg.
    """
    import subprocess

    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-ac', '1',  # mono
        '-ar', '16000',  # 16kHz
        output_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        print("ffmpeg not found. Please install ffmpeg.")
        return False


def create_streaming_manifest(audio_path: str) -> str:
    """
    Create a simple manifest file for NeMo streaming inference.
    """
    manifest_path = audio_path + ".json"

    import json
    with open(manifest_path, 'w') as f:
        json.dump({
            "audio_filepath": audio_path,
            "offset": 0,
            "duration": None,  # Will be determined automatically
            "label": "infer",
            "text": "-",  # Placeholder for ground truth
            "num_speakers": None,
            "rttm_filepath": None,
            "uem_filepath": None
        }, f)

    return manifest_path


def transcribe_streaming(
    audio_path: str,
    chunk_size: str = "medium",
    batch_size: int = 1,
    output_path: Optional[str] = None
) -> str:
    """
    Transcribe audio using Nemotron streaming ASR.

    Args:
        audio_path: Path to audio file (16kHz mono WAV)
        chunk_size: "tiny" (80ms), "small" (160ms), "medium" (560ms), "large" (1120ms)
        batch_size: Batch size for inference
        output_path: Optional path to save transcription output

    Returns:
        Transcribed text
    """
    # Chunk size configurations
    chunk_configs = {
        "tiny": "[70,0]",    # 80ms - lowest latency
        "small": "[70,1]",   # 160ms
        "medium": "[70,6]",  # 560ms - balanced
        "large": "[70,13]",  # 1120ms - best accuracy
    }

    att_context_size = chunk_configs.get(chunk_size, chunk_configs["medium"])

    print(f"Loading Nemotron Streaming ASR model...")
    print(f"Chunk size: {chunk_size} (att_context_size={att_context_size})")

    # Load the model
    asr_model = nemo_asr.models.ASRModel.from_pretrained(
        model_name="nvidia/nemotron-speech-streaming-en-0.6b"
    )

    print(f"Model loaded. Transcribing: {audio_path}")

    # For streaming inference, we need to use the NeMo streaming script
    # Here's a simplified approach using the transcribe method
    try:
        # Create manifest for inference
        manifest_path = create_streaming_manifest(audio_path)

        # Run streaming transcription
        # Note: For full cache-aware streaming with chunk-by-chunk output,
        # you would use: NeMo/examples/asr/asr_cache_aware_streaming/
        # speech_to_text_cache_aware_streaming_infer.py

        # Simplified batch transcription (processes entire file at once with streaming model)
        transcription = asr_model.transcribe(
            audio=[audio_path],
            batch_size=batch_size,
            return_hypotheses=False,
        )

        # Extract text from result
        if isinstance(transcription, list) and len(transcription) > 0:
            result = transcription[0]
            if isinstance(result, str):
                result_text = result
            elif hasattr(result, 'text'):
                result_text = result.text
            else:
                result_text = str(result)
        else:
            result_text = str(transcription)

        print(f"\nTranscription Result:")
        print("=" * 60)
        print(result_text)
        print("=" * 60)

        # Save output if requested
        if output_path:
            with open(output_path, 'w') as f:
                f.write(result_text)
            print(f"\nSaved to: {output_path}")

        # Cleanup manifest
        if os.path.exists(manifest_path):
            os.remove(manifest_path)

        return result_text

    except Exception as e:
        print(f"Error during transcription: {e}")
        import traceback
        traceback.print_exc()
        return ""


def main():
    parser = argparse.ArgumentParser(
        description="NVIDIA Nemotron Streaming ASR Sample",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transcribe sample conversation audio (2:40 coffee shop ambience)
  python nemotron.py --audio conversation_sample.wav

  # Use lowest latency (80ms chunks)
  python nemotron.py --audio conversation_sample.wav --chunk-size tiny

  # Use best accuracy (1120ms chunks)
  python nemotron.py --audio conversation_sample.wav --chunk-size large

  # Auto-convert audio to 16kHz mono first
  python nemotron.py --audio input.mp3 --convert

  # Save output to file
  python nemotron.py --audio conversation_sample.wav --output transcript.txt
        """
    )

    parser.add_argument("--audio", "-a", required=True, help="Path to audio file")
    parser.add_argument("--chunk-size", "-c",
                       choices=["tiny", "small", "medium", "large"],
                       default="medium",
                       help="Chunk size (latency vs accuracy tradeoff)")
    parser.add_argument("--batch-size", "-b", type=int, default=1,
                       help="Batch size for inference")
    parser.add_argument("--output", "-o", help="Output file for transcription")
    parser.add_argument("--convert", action="store_true",
                       help="Auto-convert audio to 16kHz mono using ffmpeg")
    parser.add_argument("--check-only", action="store_true",
                       help="Only check audio file validity without transcribing")

    args = parser.parse_args()

    # Check audio file validity
    is_valid, msg, audio_info = check_audio_file(args.audio)

    if audio_info:
        print(f"Audio info: {audio_info['channels']}ch, {audio_info['sample_rate']}Hz, {audio_info['duration']:.2f}s")

    if not is_valid:
        print(f"Invalid audio: {msg}")

        if args.convert and os.path.exists(args.audio):
            converted_path = args.audio.replace(os.path.splitext(args.audio)[1], "_16k.wav")
            print(f"\nAttempting conversion: {args.audio} -> {converted_path}")

            if convert_to_16khz_mono(args.audio, converted_path):
                print(f"Conversion successful. Using: {converted_path}")
                args.audio = converted_path
            else:
                print("Conversion failed.")
                return 1
        else:
            print("\nHint: Use --convert to auto-convert, or:")
            print("  ffmpeg -i input.wav -ac 1 -ar 16000 output.wav")
            return 1

    if args.check_only:
        print("Audio file is valid for Nemotron ASR.")
        return 0

    # Run transcription
    result = transcribe_streaming(
        audio_path=args.audio,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        output_path=args.output
    )

    return 0 if result else 1


if __name__ == "__main__":
    exit(main())

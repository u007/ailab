"""Transcribe audio using parakeet-mlx (Apple Silicon optimized)."""

import sys
from parakeet_mlx import from_pretrained


def main():
    if len(sys.argv) < 2:
        print("Usage: python parakeet_transcribe.py <audio_file> [output_file]")
        sys.exit(1)

    audio_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "output_parakeet.txt"

    print(f"Loading model...")
    model = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")

    print(f"Transcribing {audio_file}...")
    result = model.transcribe(audio_file)

    with open(output_file, "w") as f:
        for sentence in result.sentences:
            line = f"[{sentence.start:.1f}s - {sentence.end:.1f}s] ({sentence.confidence:.0%}) {sentence.text}\n"
            print(line, end="")
            f.write(line)

    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()

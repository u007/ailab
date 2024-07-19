from pydub import AudioSegment
import io
from dotenv import load_dotenv
load_dotenv()

import os
from pyannote.audio import Pipeline
import whisper

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.0",
    use_auth_token=os.environ['HF_TOKEN'])

def extract_audio_chunk_to_file_object(input_file, start_time, end_time, output_file):
    """
    Extract a chunk of audio from an audio file and save it to a file.
    
    :param input_file: Path to the input audio file.
    :param start_time: Start time in milliseconds.
    :param end_time: End time in milliseconds.
    :param output_file: Path to the output audio file.
    :return: Path to the output audio file.
    """
    # Load audio file
    audio = AudioSegment.from_file(input_file)
    
    # Extract chunk
    chunk = audio[start_time:end_time]

    chunk.export(output_file, format="wav")
    return output_file

# Load Whisper model
model = whisper.load_model("base")

audio_file = "Voice.mp3"
# apply pretrained pipeline
diarization = pipeline(audio_file, min_speakers=2, max_speakers=4)

# print the result
with open("output6.txt", "w") as file_obj:
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        tmp_file = extract_audio_chunk_to_file_object(audio_file, turn.start * 1000, turn.end * 1000, "temp.wav")
        if turn.end - turn.start < 1:
            continue
        
        # Use Whisper for transcription
        result = model.transcribe(tmp_file)
        text = result["text"]
        
        output_line = f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}: {text}\n"
        print(output_line, end='')
        file_obj.write(output_line)

# Clean up temporary file
os.remove("temp.wav")
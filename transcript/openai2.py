from dotenv import load_dotenv
load_dotenv()

from pydub import AudioSegment
import io
import os
import openai
from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.0",
    use_auth_token=os.environ['HF_TOKEN'])

def extract_audio_chunk_to_file_object(input_file, start_time, end_time, output_file):
    """
    Extract a chunk of audio from an audio file and return it as a file-like object.
    
    :param input_file: Path to the input audio file.
    :param start_time: Start time in milliseconds.
    :param end_time: End time in milliseconds.
    :param output_file: Path to the output audio file.
    :return: File-like object containing the audio chunk.
    """
    # Load audio file
    audio = AudioSegment.from_file(input_file)
    
    # Extract chunk
    chunk = audio[start_time:end_time]

    chunk.export(output_file, format="wav")
    return output_file
    
    # Create a file-like object in memory
    # buffer = io.BytesIO()
    # # Export chunk to the file-like object
    # chunk.export(buffer, format="wav")  # you can change "wav" to other formats like "mp3" if needed
    # buffer.seek(0)  # reset pointer to the beginning of the buffer
    
    # return buffer

# send pipeline to GPU (when available)
# import torch
# pipeline.to(torch.device("cuda"))

audio_file = "input_K4acryty.wav"
# apply pretrained pipeline
diarization = pipeline(audio_file, min_speakers=2, max_speakers=4)
file_obj = open("output.txt", "w")
# print the result
for turn, _, speaker in diarization.itertracks(yield_label=True):
    tmp_file = extract_audio_chunk_to_file_object(audio_file, turn.start * 1000, turn.end * 1000, "temp.wav")
    chunk_io = open(tmp_file, "rb")    
    if turn.end - turn.start < 1:
        # print(f"start={turn.start:.1f}s stop={turn.end:.1f}s too short")
        continue
    t = openai.Audio.translate("whisper-1", chunk_io)
    text = t.text
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}: {text}")
    file_obj.write(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}: {text}\n")
    # print(text)
    # file_obj.close()

file_obj.close()

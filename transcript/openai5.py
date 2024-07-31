from pydub import AudioSegment
from dotenv import load_dotenv
import time

load_dotenv()

import os
import openai
from pyannote.audio import Pipeline
from huggingface_hub import snapshot_download
import torch

# Download the model
model_id = "pyannote/speaker-diarization-3.0"
cache_dir = "./models"  # You can change this to your preferred directory

# Function to check if model is already downloaded
def is_model_downloaded(model_id, cache_dir):
    model_dir = os.path.join(cache_dir, 'models--'+ model_id.replace("/", "--"))
    return os.path.isdir(model_dir) and len(os.listdir(model_dir)) > 0

# Download the model if not already present
if not is_model_downloaded(model_id, cache_dir):
    print(f"Downloading model {model_id}...")
    snapshot_download(repo_id=model_id, cache_dir=cache_dir, use_auth_token=os.environ['HF_TOKEN'])
    print("Model downloaded successfully.")
else:
    print(f"Model {model_id} already downloaded.")

print("loading transcript model")
# Load the pipeline using the model_id
pipeline = Pipeline.from_pretrained(model_id, use_auth_token=os.environ['HF_TOKEN'])

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

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.0",
    use_auth_token=os.environ['HF_TOKEN'])

# send pipeline to GPU (when available)
# import torch
# pipeline.to(torch.device("cuda"))

audio_file = "Voice.mp3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline.to(device)

print("using", { device })
audio_file = "Voice.mp3"

print("diaration setting up...")
# apply pretrained pipeline
start_time = time.time()
diarization = pipeline(audio_file, min_speakers=2, max_speakers=2)
duration = time.time() - start_time

print(f"Pipeline execution took {duration:.4f} seconds.")
# print the result
print("processing transcript...")
start_time = time.time()
file_obj = open("output5.txt", "w")
for turn, _, speaker in diarization.itertracks(yield_label=True):
    tmp_file = extract_audio_chunk_to_file_object(audio_file, turn.start * 1000, turn.end * 1000, "temp.wav")
    if turn.end - turn.start < 1:
        # print(f"start={turn.start:.1f}s stop={turn.end:.1f}s too short")
        continue
    chunk_io = open(tmp_file, "rb")
    t = openai.Audio.translate("whisper-1", chunk_io)
    text = t.text
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}: {text}")
    file_obj.write(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}: {text}\n")
    # print(text)
    # file_obj.close()
file_obj.close()

duration = time.time() - start_time

print(f"done whisper translation took {duration:.4f} seconds.")
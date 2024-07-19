from pydub import AudioSegment
import io
from dotenv import load_dotenv
load_dotenv()

import os
from pyannote.audio import Pipeline
import whisper
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

# Load Whisper model
model = whisper.load_model("base")

# Send pipeline to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline.to(device)

audio_file = "Voice.mp3"
# apply pretrained pipeline
diarization = pipeline(audio_file, min_speakers=2, max_speakers=4)

# print the result
with open("output5.txt", "w") as file_obj:
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
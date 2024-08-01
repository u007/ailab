# `pip3 install assemblyai` (macOS)
# `pip install assemblyai` (Windows)

import assemblyai as aai
from dotenv import load_dotenv
load_dotenv()
import os
aai.settings.api_key = os.environ['ASSEMBLYAI_API_KEY']
# transcriber = aai.Transcriber()
transcriber = aai.Transcriber()

file1 = "https://montespic.sgp1.digitaloceanspaces.com/chua/mc3.mp3"
file2 = "https://montespic.sgp1.digitaloceanspaces.com/chua/Voice.mp3"
file3 = "https://montespic.sgp1.digitaloceanspaces.com/chua/input_K4acryty.wav"

config = aai.TranscriptionConfig(speaker_labels=True)

print("processing transcript...")
transcript = transcriber.transcribe(file3, config=config)
# transcript = transcriber.transcribe("./my-local-audio-file.wav")
# print(transcript.text)
file_obj = open("output10-file3.txt", "w")
for utterance in transcript.utterances:
  print(f"Speaker {utterance.speaker}: {utterance.text}")
  file_obj.write(f"speaker_{utterance.speaker}: {utterance.text}\n")

file_obj.close()

print("done.")
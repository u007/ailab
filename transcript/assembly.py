# `pip3 install assemblyai` (macOS)
# `pip install assemblyai` (Windows)

import assemblyai as aai

aai.settings.api_key = "4aa5c9989ce0499f9291b6d15223000b"
# transcriber = aai.Transcriber()
transcriber = aai.Transcriber()


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
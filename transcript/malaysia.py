from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
from datasets import Audio
# import requests
import soundfile as sf

sr = 16000
audio = Audio(sampling_rate=sr)

audio_file = "Voice.mp3"

processor = AutoProcessor.from_pretrained("mesolitica/malaysian-whisper-base")
model = AutoModelForSpeechSeq2Seq.from_pretrained("mesolitica/malaysian-whisper-base")


y, sr = sf.read(audio_file)
inputs = processor([y], sampling_rate=sr, return_tensors='pt')
r = model.generate(inputs['input_features'], language='ms', return_timestamps=True)
processor.tokenizer.decode(r[0])

print(r[0])
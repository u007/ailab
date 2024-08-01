from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
from datasets import Audio
import math
# import requests
import soundfile as sf
import re

sr = 16000
audio = Audio(sampling_rate=sr)

audio_file = "Voice16k.wav"

processor = AutoProcessor.from_pretrained("mesolitica/malaysian-whisper-base")
model = AutoModelForSpeechSeq2Seq.from_pretrained("mesolitica/malaysian-whisper-base")
y, sr = sf.read(audio_file)

chunk_duration = 20  # seconds
samples_per_chunk = sr * chunk_duration

for i in range(0, len(y), samples_per_chunk):
    chunk = y[i:i+samples_per_chunk]
    
    inputs = processor([chunk], sampling_rate=sr, return_tensors='pt')
    r = model.generate(inputs['input_features'], language='en', return_timestamps=True)
    result = processor.tokenizer.decode(r[0])
    
    # Remove duplicated text using regular expressions
    result = re.sub(r'(.)\1\1+', r'\1\1', result)
    result = re.sub(r'\b(\w+\b(?:, \b\w+\b)*)\b(?:, \1\b)+', r'\1', result)
    
    print(f"Chunk {math.floor(i/samples_per_chunk) + 1} result:")
    print(result)
    print()
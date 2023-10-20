from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import AutoDetectDecodingConfig, RecognizeRequest
from google.cloud.speech_v2.types import RecognitionConfig
from google.cloud.speech_v2.types import RecognitionFeatures
from google.cloud.speech_v2.types import SpeakerDiarizationConfig

import wave

def get_sample_rate(audio_file):
    with wave.open(audio_file, "rb") as wav:
        return wav.getframerate()

# import print_env
# with wave.open(wav_file, "rb") as wav:
#     num_channels = wav.getnchannels() 
#     sample_width = wav.getsampwidth()
#     frame_rate = wav.getframerate()
#     num_frames = wav.getnframes()

#     print(f"Number of channels: {num_channels}")
#     print(f"Sample width: {sample_width} bytes") 
#     print(f"Frame rate: {frame_rate} Hz")
#     print(f"Number of frames: {num_frames}")
# client = speech.SpeechClient()
credentials_path = "gcloud.json"
client = SpeechClient()

# audio_path = "mc1.wav"
audio_path = "mc2.wav"

# Configure the speaker diarization
diarization_config = SpeakerDiarizationConfig(min_speaker_count=3, max_speaker_count=3)
sample_rate = get_sample_rate(audio_path)
# Configure the recognition
config = RecognitionConfig(
    auto_decoding_config=AutoDetectDecodingConfig(),
    language_codes=["en-US"],
    # https://cloud.google.com/speech-to-text/v2/docs/transcription-model
    model="long",
    # https://cloud.google.com/python/docs/reference/speech/latest/google.cloud.speech_v2.types.RecognitionFeatures
    features=RecognitionFeatures(
        enable_automatic_punctuation=True,
        diarization_config=diarization_config,
    ),
)

# Read the audio file
with open(audio_path, "rb") as audio_file:
    content = audio_file.read()

request = RecognizeRequest(
    recognizer=f"projects/movingfwd-a662b/locations/global/recognizers/_",
    config=config,
    content=content,
)

# Transcribes the audio into text
response = client.recognize(request=request)

for result in response.results:
    print(f"Transcript: {result.alternatives[0].transcript}")

# # Perform the speech recognition
# response = client.recognize(config=recognition_config, audio={"content": content})

# print("results: ", response.results)
# # Print the results
# result = response.results[-1]
# words_info = result.alternatives[0].words

# # Printing out the output:
# last_speaker = -1
# current_words = ""
# for word_info in words_info:
#     if last_speaker > -1 and last_speaker != word_info.speaker_tag:
#         print(f"Speaker {last_speaker}: {current_words}")
#         current_words = word_info.word
#     else:
#         current_words += " " + word_info.word
    
#     last_speaker = word_info.speaker_tag

# if current_words != "":
#     print(f"Speaker {last_speaker}: {current_words}")
# from google.cloud import speech
from google.cloud import speech_v1p1beta1 as speech
import wave

def get_sample_rate(audio_file):
    with wave.open(audio_file, "rb") as wav:
        return wav.getframerate()

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
client = speech.SpeechClient.from_service_account_json(credentials_path)

audio_path = "mc1.wav"

# Configure the speaker diarization
diarization_config = speech.SpeakerDiarizationConfig(enable_speaker_diarization=True, min_speaker_count=3, max_speaker_count=3)
sample_rate = get_sample_rate(audio_path)
# Configure the recognition
recognition_config = speech.RecognitionConfig(
    # encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=sample_rate,
    language_code="en-US",
    enable_automatic_punctuation=False,
    diarization_config=diarization_config  # Add the diarization config to the recognition config
)

# Read the audio file
with open(audio_path, "rb") as audio_file:
    content = audio_file.read()

# Perform the speech recognition
response = client.recognize(config=recognition_config, audio={"content": content})

# print("results: ", response.results)
# Print the results
result = response.results[-1]
words_info = result.alternatives[0].words

# Printing out the output:
last_speaker = -1
current_words = ""
for word_info in words_info:
    if last_speaker > -1 and last_speaker != word_info.speaker_tag:
        print(f"Speaker {last_speaker}: {current_words}")
        current_words = word_info.word
    else:
        current_words += " " + word_info.word
    
    last_speaker = word_info.speaker_tag

if current_words != "":
    print(f"Speaker {last_speaker}: {current_words}")
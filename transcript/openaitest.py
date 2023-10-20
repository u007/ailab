
import openai
audio_file= open("mc2.wav", "rb")

# https://platform.openai.com/docs/guides/speech-to-text/quickstart
transcript = openai.Audio.transcribe("whisper-1", audio_file)

# translate
# transcript = openai.Audio.translate("whisper-1", audio_file)
print(f"Transcript: {transcript}")

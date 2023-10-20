import speech_recognition as sr
from pydub import AudioSegment

def convert_mp3_to_wav(input_mp3):
    # Load the MP3 file using pydub
    audio = AudioSegment.from_mp3(input_mp3)

    output_wav = input_mp3.split('.')[0] + '.wav'

    # Export the audio to WAV format
    audio.export(output_wav, format="wav")
    return output_wav

def transcribe_audio(audio_file):
    # Initialize the recognizer
    recognizer = sr.Recognizer()
    if audio_file.endswith(".mp3"):
        wav_file = convert_mp3_to_wav(audio_file)
    else:
        wav_file = audio_file

    print("wave file: ", wav_file)
    # Load the audio file
    with sr.AudioFile(wav_file) as audio_source:
        # Adjust for ambient noise if necessary
        recognizer.adjust_for_ambient_noise(audio_source)

        # Recognize speech from the audio file
        audio = recognizer.record(audio_source)

        # Use the Google Web Speech API for transcription
        try:
            # https://groups.google.com/access-error?continue=https://groups.google.com/g/cloud-speech-discuss/c/C7KIJIAPt68
            transcript = recognizer.recognize_google(audio)
            return transcript
        except sr.UnknownValueError:
            return "Unable to recognize speech."
        except sr.RequestError as e:
            return f"Error: {e}"

if __name__ == "__main__":
    audio_file_path = "mc1.mp3"
    # audio_file_path = "mc2.1.mp3"
    # audio_file_path = "mc3.mp3"

    transcript = transcribe_audio(audio_file_path)
    print("Transcript:")
    print(transcript)
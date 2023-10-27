from pydub import AudioSegment

def convert_mkv_to_wav(input_file, output_file):
    """
    Convert an MKV file to a WAV file.
    
    :param input_file: Path to the input MKV file.
    :param output_file: Path to the output WAV file.
    """
    # Load audio file
    audio = AudioSegment.from_file(input_file)
    
    # Export audio chunk to WAV file
    audio.export(output_file, format="wav")
    return output_file

# from moviepy.editor import VideoFileClip
# def convert_mkv_to_wav(input, output):
#     """
#     Convert an MKV file to a WAV file.
    
#     :param input: Path to the input MKV file.
#     :param output: Path to the output WAV file.
#     """
#     # Load the video clip
#     video_clip = VideoFileClip(input)

#     # Extract the audio
#     audio_clip = video_clip.audio

#     # Write the audio to a WAV file
#     audio_clip.write_audiofile(output)

#     # Close the audio and video clips
#     audio_clip.close()
#     video_clip.close()

#     return output


import os
import subprocess
import requests

def generate_audio_from_text(api_key, voice_id, text, output_file):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "Content-Type": "application/json",
        "xi-api-key": api_key
    }
    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",  # or another model_id based on your requirement
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        with open(output_file, 'wb') as f:
            f.write(response.content)
    else:
        print("Error:", response.status_code, response.text)

# Example usage
generate_audio_from_text('<your_xi-api-key>', '<chosen_voice_id>', 'Your text here', 'output.mp3')


def create_video_from_image(image_path, duration, output_video_path):
    """
    Creates a video from an image file using FFmpeg.

    Args:
    image_path (str): Path to the image file.
    duration (int): Duration of the video in seconds.
    output_video_path (str): Path where the output video will be saved.
    """
    cmd = f"ffmpeg -loop 1 -i {image_path} -c:v libx264 -t {duration} -pix_fmt yuv420p {output_video_path}"
    subprocess.run(cmd, shell=True)

def run_wav2lip(checkpoint_path, face_video_path, audio_file_path, output_path, pads=None, resize_factor=None):
    """
    Runs the Wav2Lip model to lip-sync a video to an audio file.

    Args:
    checkpoint_path (str): Path to the Wav2Lip model weights.
    face_video_path (str): Path to the face video file.
    audio_file_path (str): Path to the audio file.
    output_path (str): Path where the lip-synced video will be saved.
    pads (tuple): Padding for the face bounding box (top, bottom, left, right).
    resize_factor (int): Factor to resize the video resolution.
    """
    cmd = f"python inference.py --checkpoint_path {checkpoint_path} --face {face_video_path} --audio {audio_file_path} --outfile {output_path}"
    if pads:
        pad_str = ' '.join(map(str, pads))
        cmd += f" --pads {pad_str}"
    if resize_factor:
        cmd += f" --resize_factor {resize_factor}"
    
    subprocess.run(cmd, shell=True)

# Example usage:
image_path = 'images\image'
duration = 30  # Duration in seconds
output_video_path = 'path/to/output/video.mp4'

create_video_from_image(image_path, duration, output_video_path)

# Wav2Lip parameters
checkpoint_path = 'path/to/wav2lip/weights.pth'
face_video_path = output_video_path
audio_file_path = 'path/to/your/audio.mp3'
output_path = 'path/to/lipsync/output.mp4'
pads = (0, 10, 0, 0)  # Example padding values
resize_factor = 2  # Example resize factor

run_wav2lip(checkpoint_path, face_video_path, audio_file_path, output_path, pads, resize_factor)

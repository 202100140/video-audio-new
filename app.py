import streamlit as st
import openai
import whisper
from moviepy.editor import VideoFileClip, AudioFileClip
from gtts import gTTS
import tempfile
import requests
import os


def main():
    st.title("Video Audio Replacement with Whisper and GPT-4o")

    # Azure OpenAI connection details
    azure_openai_key = "22ec84421ec24230a3638d1b51e3a7dc"
    azure_openai_endpoint = "https://internshala.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview"

    # Load Whisper model
    model = whisper.load_model("base")

    # File upload section
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov"])

    # Check if a file was uploaded
    if uploaded_file:
        st.video(uploaded_file)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video_file:
            temp_video_file.write(uploaded_file.read())
            video_clip = VideoFileClip(temp_video_file.name)
            audio_clip = video_clip.audio
            st.write("Extracted audio from video.")

            # Save audio to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio_file:
                audio_clip.write_audiofile(temp_audio_file.name)
                st.write(f"Audio extracted to: {temp_audio_file.name}")

                # Transcribe the audio with Whisper
                st.write("Transcribing audio using Whisper...")
                transcription_result = model.transcribe(temp_audio_file.name)
                transcription = transcription_result["text"]
                st.write("Transcription completed:", transcription)

                # Correct transcription with GPT-4o
                if st.button("Correct Transcription with GPT-4o"):
                    if azure_openai_key and azure_openai_endpoint:
                        try:
                            # Setting up headers for the API request
                            headers = {
                                "Content-Type": "application/json",
                                "api-key": azure_openai_key
                            }

                            # Data for GPT-4o API request
                            data = {
                                "messages": [{"role": "user",
                                              "content": f"Correct the following transcription: {transcription}"}],
                                "max_tokens": 1000
                            }

                            # Make the POST request to the GPT-4o endpoint
                            response = requests.post(azure_openai_endpoint, headers=headers, json=data)

                            if response.status_code == 200:
                                result = response.json()
                                corrected_transcription = result["choices"][0]["message"]["content"].strip()
                                st.success("Corrected Transcription: " + corrected_transcription)

                                # Generate new audio using gTTS
                                st.write("Generating AI voice from corrected transcription...")
                                tts = gTTS(corrected_transcription, lang='en')

                                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_tts_audio_file:
                                    tts.save(temp_tts_audio_file.name)
                                    st.write(f"New audio generated at: {temp_tts_audio_file.name}")

                                    # Replace audio in video
                                    new_audio = AudioFileClip(temp_tts_audio_file.name)
                                    final_video = video_clip.set_audio(new_audio)

                                    # Save final video to a file
                                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_output_file:
                                        final_video.write_videofile(temp_output_file.name, codec="libx264",
                                                                    audio_codec="aac")
                                        st.success("Video with new audio generated successfully.")
                                        st.video(temp_output_file.name)
                            else:
                                st.error(f"Failed to get GPT-4o response: {response.status_code} - {response.text}")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                    else:
                        st.warning("Please provide Azure OpenAI key and endpoint.")


if __name__ == "__main__":
    main()

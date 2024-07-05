import streamlit as st
from streamlit_mic_recorder import mic_recorder
import soundfile as sf

st.write("Record your voice, and play the recorded audio:")
audio = mic_recorder(start_prompt="⏺️", stop_prompt="⏹️", key='recorder')

if audio:
    st.audio(audio['bytes'])

    # Convert the recorded audio to FLAC and save it
    flac_file_path = "recorded_audio.flac"
    with open(flac_file_path, 'wb') as flac_file:
        flac_file.write(audio['bytes'])

    st.success(f"Audio recorded and saved as {flac_file_path}")

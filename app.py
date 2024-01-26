import streamlit as st
from transformers import pipeline
import torchaudio
from io import BytesIO
# Assuming st_audiorec is defined elsewhere
from st_audiorec import st_audiorec

@st.cache
def load_data_HEA_exp():
    global stt_pipeline
    stt_pipeline = pipeline("automatic-speech-recognition", model="Salama1429/KalemaTech-Arabic-STT-ASR-based-on-Whisper-Small")
    #return pipeline("automatic-speech-recognition", model="Salama1429/KalemaTech-Arabic-STT-ASR-based-on-Whisper-Small")



def transcribe_audio(audio_array):

    transcription = stt_pipeline(audio_array)

    return transcription

st.title("QuranPal 🕋")

wav_audio_data = st_audiorec()



#uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav"])

if wav_audio_data is not None:
    # st.audio(wav_audio_data, format="audio")

    if st.button("Transcribe"):
        st.write("Transcribing...")

        # Assuming you have the correct audio data type here
        transcription = transcribe_audio(wav_audio_data)
        
        # Display the transcription
        st.subheader("Transcription:")
        st.write(transcription["text"])

Python 3.11.9 (tags/v3.11.9:de54cf5, Apr  2 2024, 10:12:12) [MSC v.1938 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> import streamlit as st
... import speech_recognition as sr
... from pydub import AudioSegment
... def convert_audio_to_wav(audio_file):
... audio = AudioSegment.from_file(audio_file)
... wav_file = audio_file.name.split(".")[0] + ".wav"
... audio.export(wav_file, format="wav")
... return wav_file
... def speech_to_text(audio_file):
... recognizer = sr.Recognizer()
... with sr.AudioFile(audio_file) as source:
... audio = recognizer.record(source)
... try:
... text = recognizer.recognize_google(audio)
... return text
... except sr.UnknownValueError:
... return "Could not understand audio"
... except sr.RequestError as e:
... return f"Error: {str(e)}"
... def main():
... st.title("Speech to Text Converter")
... st.write("Upload an audio file and convert it to text.")
... uploaded_file = st.file_uploader("Choose an audio file", type=["wav",
... "mp3"])
... if uploaded_file is not None:
... file_details = {"Filename": uploaded_file.name, "FileType":
... uploaded_file.type}
... st.write(file_details)
... if uploaded_file.type == "audio/mp3":
... uploaded_file = convert_audio_to_wav(uploaded_file)
... text = speech_to_text(uploaded_file)
... st.write("Converted Text:")
... st.write(text)
... if _name_ == "_main_":

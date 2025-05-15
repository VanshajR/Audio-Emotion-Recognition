import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import tempfile
from audio_processor import AudioProcessor
import soundfile as sf
import pickle
from tensorflow.keras.models import load_model
from pydub import AudioSegment

# Paths to model and scaler
MODEL_PATH = "models/emotion_model.h5"
SCALER_PATH = "models/scaler.pkl"

# Load model and scaler
try:
    emotion_model_loaded = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading model: {e}")
    emotion_model_loaded = None

try:
    with open(SCALER_PATH, "rb") as f:
        scaler_loaded = pickle.load(f)
except Exception as e:
    st.error(f"Error loading scaler: {e}")
    scaler_loaded = None

# Define the emotion labels (must match training)
EMOTIONS = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

def predict_emotion(features):
    if emotion_model_loaded is None or scaler_loaded is None:
        raise ValueError("Model or scaler not loaded.")
    features_scaled = scaler_loaded.transform(features.reshape(1, -1))
    prediction = emotion_model_loaded.predict(features_scaled)[0]
    emotion_idx = prediction.argmax()
    confidence = prediction[emotion_idx]
    return EMOTIONS[emotion_idx], confidence

def predict_proba(features):
    if emotion_model_loaded is None or scaler_loaded is None:
        raise ValueError("Model or scaler not loaded.")
    features_scaled = scaler_loaded.transform(features.reshape(1, -1))
    prediction = emotion_model_loaded.predict(features_scaled)[0]
    return list(zip(EMOTIONS, prediction))

# Set page config
st.set_page_config(
    page_title="Audio Emotion Recognition",
    page_icon="🎵",
    layout="wide"
)

# Initialize audio processor
audio_processor = AudioProcessor()

def plot_waveform(audio, sr):
    plt.figure(figsize=(8, 3))  # Reduced size
    librosa.display.waveshow(audio, sr=sr)
    plt.title('Audio Waveform')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    return plt

def plot_spectrogram(audio, sr):
    plt.figure(figsize=(8, 3))  # Reduced size
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    return plt

def convert_to_wav(input_file, output_path):
    """Convert any audio file to WAV format."""
    try:
        audio = AudioSegment.from_file(input_file)
        audio.export(output_path, format="wav")
        return True
    except Exception as e:
        st.error(f"Error converting audio file: {e}")
        return False

def main():
    st.title("🎵 Audio Emotion Recognition")
    st.write("Analyze emotions from speech using AI")

    st.header("Upload Audio")
    uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'ogg', 'flac', 'm4a', 'aac'])

    if uploaded_file is not None:
        # Create a temporary file for the uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_input:
            temp_input.write(uploaded_file.getvalue())
            temp_input_path = temp_input.name

        # Create a temporary WAV file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_wav:
            temp_wav_path = temp_wav.name

        # Convert to WAV if needed
        if uploaded_file.name.lower().endswith('.wav'):
            temp_wav_path = temp_input_path
        else:
            if not convert_to_wav(temp_input_path, temp_wav_path):
                st.error("Failed to convert audio file to WAV format.")
                os.unlink(temp_input_path)
                return

        # Display the audio player
        st.audio(uploaded_file, format='audio/wav')
        
        # Process the WAV file
        display_results(temp_wav_path)
        
        # Clean up temporary files
        os.unlink(temp_input_path)
        if temp_wav_path != temp_input_path:
            os.unlink(temp_wav_path)
    else:
        st.info("Please upload an audio file to analyze.")

def display_results(audio_path):
    # Create a placeholder for the loading spinner
    with st.spinner('Processing audio and analyzing emotions...'):
        audio, features = audio_processor.process_audio_file(audio_path)
        
        if audio is not None and features is not None:
            try:
                # Show loading spinner while making predictions
                with st.spinner('Making predictions...'):
                    emotion, confidence = predict_emotion(features)
                    probs = predict_proba(features)
                
                # Display emotion result prominently
                st.markdown("---")
                st.markdown(f"## 🎯 Detected Emotion: {emotion.upper()}")
                st.markdown(f"### Confidence: {confidence:.2%}")
                st.markdown("---")
                
                # Create two columns for plots
                col1, col2 = st.columns(2)
                with col1:
                    st.pyplot(plot_waveform(audio, audio_processor.sample_rate))
                with col2:
                    st.pyplot(plot_spectrogram(audio, audio_processor.sample_rate))
                
                # Display probability distribution
                st.subheader("Emotion Probability Distribution")
                fig, ax = plt.subplots(figsize=(10, 3))  # Reduced height
                emotions, probabilities = zip(*probs)
                ax.bar(emotions, probabilities)
                ax.set_title("Emotion Probability Distribution")
                ax.set_ylim(0, 1)
                plt.xticks(rotation=45)
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Prediction error: {e}")
        else:
            st.error("Error processing audio file. Please try again.")

if __name__ == "__main__":
    main() 
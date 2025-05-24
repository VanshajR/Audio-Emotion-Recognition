# 🎵 Audio Emotion Recognition

A powerful web application that uses machine learning to analyze and detect emotions from speech audio files. The application is built with Streamlit and provides real-time emotion analysis with visual representations of audio features.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://audio-sentiment-vanshajr.streamlit.app/)

## ✨ Features

- **Multi-format Support**: Upload audio files in various formats (WAV, MP3, OGG, FLAC, M4A, AAC)
- **Real-time Analysis**: Instant emotion detection from uploaded audio
- **Visual Analytics**:
  - Audio waveform visualization
  - Spectrogram analysis
  - Emotion probability distribution chart
- **Detailed Results**:
  - Primary emotion detection with confidence score
  - Probability distribution across all emotions
  - Support for 8 different emotions: angry, calm, disgust, fearful, happy, neutral, sad, surprised

## 🚀 Live Demo

Try the application live at: [https://audio-sentiment-vanshajr.streamlit.app/](https://audio-sentiment-vanshajr.streamlit.app/)

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **Audio Processing**: Librosa, SoundFile
- **Machine Learning**: TensorFlow/Keras
- **Data Visualization**: Matplotlib
- **Audio Conversion**: Pydub

## 📋 Prerequisites

- Python 3.7+
- pip (Python package installer)

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/audio-emotion-recognition.git
cd audio-emotion-recognition
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure you have the model files in the correct location:
   - Place `emotion_model.h5` in the `models/` directory
   - Place `scaler.pkl` in the `models/` directory

## 🎮 Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)

3. Upload an audio file using the file uploader

4. View the analysis results:
   - Detected emotion and confidence score
   - Audio waveform visualization
   - Spectrogram
   - Emotion probability distribution

## 📝 Project Structure

```
audio-emotion-recognition/
├── app.py                 # Main Streamlit application
├── audio_processor.py     # Audio processing utilities
├── models/               # Model directory
│   ├── emotion_model.h5  # Trained emotion detection model
│   └── scaler.pkl        # Feature scaler
└── requirements.txt      # Project dependencies
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📜 Acknowledgments

- The project uses the RAVDESS dataset for emotion recognition
- Thanks to all the open-source libraries that made this project possible 
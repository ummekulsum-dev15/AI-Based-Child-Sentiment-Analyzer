# Child Sentiment and Emotion Analyzer AI Project

A simple AI project that uses TextBlob and Gradio to analyze the sentiment of a sentence.

## 🚀 Quick Start
```bash
pip install -r requirements.txt
python main.py
```

## 📁 Repository Contents
- `Correct_Child_Sentiment_and_Emotion_Analyzer_AI.ipynb` - Jupyter notebook with the sentiment analyzer demo
- `main.py` - Web UI entrypoint using Gradio
- `requirements.txt` - Python dependencies
- `README.md` - Project documentation

## 🧠 How It Works
1. Enter text into the Gradio input box, or upload audio/video.
2. The app uses `TextBlob` for text sentiment, a speech emotion model for audio, and face-emotion detection for video.
3. It returns labels such as `Positive`, `Negative`, `Neutral`, `Happy`, `Sad`, `Angry`, or `Neutral`.

## 📦 Requirements
- Python 3.8+
- `textblob`
- `gradio`
- `transformers`
- `torch`
- `librosa`
- `soundfile`
- `opencv-python`
- `deepface`
- `tensorflow`
- `numpy`

## ✨ Notes
- This repository currently contains the notebook demo and a simple Gradio app.
- The notebook and app both use TextBlob for sentiment classification.

## 📄 License
This project is licensed under the MIT License if a `LICENSE` file is present.

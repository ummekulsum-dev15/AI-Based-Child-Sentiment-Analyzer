# Child Sentiment and Emotion Analyzer AI - Complete Guide

This guide explains the trained emotion recognition model in your Child Sentiment and Emotion Analyzer AI.

## ✅ What's Already Done

Your project now includes:

1. ✅ **Baby cry dataset** (3000 files from Kaggle)
2. ✅ **Trained Fast ML Model** (Random Forest, 55% accuracy)
3. ✅ **Integration with main.py** (automatic model loading)
4. ✅ **Training scripts** for retraining if needed

## 🚀 Quick Start (Use the Model)

Just run your app - it will automatically use the trained baby cry model:

```bash
.venv\Scripts\python.exe main.py
```

The app will show "✅ **Using trained baby cry model** for better accuracy" in the Audio tab.

## 📊 Model Details

**Model Type:** Random Forest Classifier  
**Training Data:** 3000 baby audio files from Kaggle dataset  
**Accuracy:** ~55% overall  
**Training Time:** ~30 seconds (fast ML model)

### Emotion Mapping

| Dataset Folder | Model Label | App Display | Files |
|---------------|-------------|-------------|-------|
| hungry | hunger | Hunger | 382 |
| discomfort | irritated | Irritated | 138 |
| scared | fear | Fear | 27 |
| silence | normal | Normal | 108 |
| tired | sad | Sad | 136 |
| laugh | happy | Normal | 108 |

## 📁 Project Structure

```
Sentiment-Analyzer-AI/
├── baby_cry_dataset/              # Organized dataset (2000 files)
│   ├── hunger/                    # 382 files
│   ├── irritated/                 # 138 files
│   ├── fear/                      # 27 files
│   ├── normal/                    # 108 files
│   ├── sad/                       # 136 files
│   ├── happy/                     # 108 files
│   └── metadata.csv
├── baby_emotion_model_fast/       # Trained model
│   ├── baby_cry_classifier.pkl    # Random Forest model
│   └── metadata.pkl               # Model metadata
├── prepare_dataset.py             # Dataset download/organization script
├── train_model_fast.py            # Fast model training script
├── test_model_fast.py             # Model testing script
├── main.py                        # Main app (uses trained model automatically)
└── TRAINING_GUIDE.md              # This file
```

## 🔄 Retraining the Model

If you want to retrain with more data or different settings:

```bash
# Step 1: Prepare dataset (downloads from Kaggle)
.venv\Scripts\python.exe prepare_dataset.py

# Step 2: Train model (takes ~30 seconds)
.venv\Scripts\python.exe train_model_fast.py

# Step 3: Test model
.venv\Scripts\python.exe test_model_fast.py baby_cry_dataset/hunger

# Step 4: Run app (uses new model automatically)
.venv\Scripts\python.exe main.py
```

## 🧪 Testing the Model

Test on individual files or folders:

```bash
# Test single file
.venv\Scripts\python.exe test_model_fast.py "baby_cry_dataset\hunger\file.wav"

# Test shows detailed predictions with confidence scores
```

## 📈 Model Performance

### Per-Emotion Accuracy

| Emotion | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| Hunger | 51% | 64% | 57% |
| Irritated | 12% | 7% | 9% |
| Fear | 80% | 80% | 80% |
| Normal | 100% | 100% | 100% |
| Sad | 5% | 4% | 4% |
| Happy | 100% | 100% | 100% |

**Overall Accuracy:** 55.56%

### Why These Numbers?

- ✅ **Normal, Happy, Fear**: Very well detected (80-100%)
- ⚠️ **Hunger**: Moderate detection (57%)
- ❌ **Irritated, Sad**: Hard to distinguish (low samples or similarity)

## 🎯 Improving Accuracy

To get better results:

1. **More Training Data**
   - Add more baby cry recordings
   - Ensure balanced samples per emotion
   - Target: 500+ samples per emotion

2. **Better Features**
   - Try deeper audio features
   - Add pitch, formants, harmonic analysis

3. **Deep Learning**
   - Use `train_model.py` for wav2vec2 fine-tuning
   - Requires GPU or long CPU time (hours)

4. **Data Augmentation**
   - Add background noise
   - Change pitch/speed
   - Create variations

## 🔧 Troubleshooting

### App Shows "Using general model"
- Make sure `baby_emotion_model_fast/` folder exists
- Run: `.venv\Scripts\python.exe train_model_fast.py`

### Low Accuracy for Specific Emotions
- Check if you have enough training samples
- Some emotions sound similar (irritated vs sad)
- Try recording more specific samples

### Module Not Found Errors
```bash
.venv\Scripts\pip.exe install librosa scikit-learn joblib
```

## 📝 Notes

- The model automatically loads when you run main.py
- Falls back to general model if trained model not found
- Training takes ~30 seconds on CPU
- Works with both file uploads and microphone recordings
- Model processes audio in real-time

## 🎓 Technical Details

### Feature Extraction
The model extracts these audio features:
- **MFCCs** (13 coefficients + stats) - most important for audio classification
- **Spectral Centroid** - brightness of sound
- **Spectral Rolloff** - frequency distribution
- **Zero Crossing Rate** - texture/noisiness
- **RMS Energy** - loudness
- **Tempo** - rhythm/speed

### Classification
- **Algorithm:** Random Forest (200 trees)
- **Advantages:** Fast, interpretable, works on CPU
- **Alternative:** wav2vec2 (deep learning, slower to train, potentially more accurate)

---

**Author:** Mahmuda Anjum Shamme
**Project:** Child Sentiment and Emotion Analyzer AI
**Dataset:** Kaggle Baby Cry Dataset (mennaahmed23/baby-cry)
**Model:** Random Forest with audio feature extraction

## 📋 Prerequisites

Make sure you have the required libraries installed:

```bash
.venv\Scripts\pip.exe install librosa datasets scikit-learn joblib
```

## 🚀 Quick Start (3 Steps)

### Step 1: Download and Prepare Dataset

```bash
.venv\Scripts\python.exe prepare_dataset.py
```

This will:
- Download the baby cry dataset from Kaggle (`mennaahmed23/baby-cry`)
- Organize files into emotion folders
- Create a metadata.csv file for training

**Folder Mapping:**
- `hungry` → `hunger`
- `discomfort` → `irritated`
- `scared` → `fear`
- `silence` → `normal`
- `tired` → `sad`
- `laughing` → `happy`

### Step 2: Train the Model

```bash
.venv\Scripts\python.exe train_model.py
```

This will:
- Load all audio files
- Preprocess them to 16kHz sample rate
- Fine-tune wav2vec2 model on baby cry data
- Save the trained model to `./baby_emotion_model/`

**Training Details:**
- **Epochs:** 20 (with early stopping)
- **Batch Size:** 8
- **Learning Rate:** 1e-4
- **Train/Test Split:** 80/20
- **Output:** `./baby_emotion_model/`

**Training Time:**
- CPU: ~30-60 minutes
- GPU: ~10-20 minutes

### Step 3: Test the Model

```bash
# Test a single file
.venv\Scripts\python.exe test_model.py baby_cry_dataset/hunger/file.wav

# Test a folder
.venv\Scripts\python.exe test_model.py baby_cry_dataset/hunger/

# Run demo tests
.venv\Scripts\python.exe test_model.py
```

### Step 4: Run Your App

```bash
.venv\Scripts\python.exe main.py
```

The app will **automatically** use the trained baby cry model if it exists in `./baby_emotion_model/`.

## 📊 Model Architecture

```
wav2vec2-lg-xlsr-en-speech-emotion-recognition
    ↓
Feature Extractor (frozen)
    ↓
Classification Head (trained)
    ↓
6 Emotion Classes: hunger, irritated, fear, normal, sad, happy
```

## 🎯 Emotion Mapping

The model predicts these emotions which map to your app's categories:

| Dataset Folder | Model Label | App Display |
|---------------|-------------|-------------|
| hungry | hunger | Hunger |
| discomfort | irritated | Irritated |
| scared | fear | Fear |
| silence | normal | Normal |
| tired | sad | Sad |
| laughing | happy | Normal |

## 🔧 Configuration

You can modify these in `train_model.py`:

```python
EMOTION_LABELS = ['hunger', 'irritated', 'fear', 'normal', 'sad', 'happy']
MODEL_NAME = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
SAMPLE_RATE = 16000
MAX_DURATION = 5  # seconds
OUTPUT_DIR = "./baby_emotion_model"
```

## 📁 File Structure

```
Sentiment-Analyzer-AI/
├── prepare_dataset.py      # Download and organize dataset
├── train_model.py          # Train the model
├── test_model.py           # Test the model
├── main.py                 # Main app (uses trained model)
├── baby_cry_dataset/       # Organized dataset (created by prepare_dataset.py)
│   ├── hunger/
│   ├── irritated/
│   ├── fear/
│   ├── normal/
│   ├── sad/
│   ├── happy/
│   └── metadata.csv
└── baby_emotion_model/     # Trained model (created by train_model.py)
    ├── pytorch_model.bin
    ├── config.json
    ├── processor files...
    └── emotion_labels.pkl
```

## 🐛 Troubleshooting

### Error: "No module named 'librosa'"
```bash
.venv\Scripts\pip.exe install librosa
```

### Error: "Model not found"
Make sure you ran `prepare_dataset.py` and `train_model.py` first.

### Training is too slow
- Use GPU if available (automatic with CUDA)
- Reduce `num_train_epochs` in train_model.py
- Reduce `per_device_train_batch_size` if out of memory

### Low accuracy
- Get more training data
- Increase training epochs
- Check if audio files are properly labeled
- Ensure audio quality is good

## 📈 Improving Accuracy

1. **More Data:** Add more baby cry samples
2. **Data Augmentation:** Add noise, pitch shifts, time stretches
3. **Longer Training:** Increase epochs (watch for overfitting)
4. **Balance Classes:** Ensure equal samples per emotion
5. **Better Audio:** Use high-quality recordings

## 🔄 Retraining

If you get more data or want to improve the model:

```bash
# Remove old model
rmdir /s baby_emotion_model

# Retrain
.venv\Scripts\python.exe train_model.py
```

## 📝 Notes

- The model automatically falls back to the general model if baby model is not found
- Training creates a model specifically tuned for baby cries
- Test accuracy should be >70% for good results
- The app will show which model is being used (baby vs general)

## 🎓 Next Steps

After training, you can:
1. Deploy the model to production
2. Share the model with others
3. Continue training with new data
4. Export to ONNX for faster inference

---

**Author:** Mahmuda Anjum Shamme
**Project:** Child Sentiment and Emotion Analyzer AI

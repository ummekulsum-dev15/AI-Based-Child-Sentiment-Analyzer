"""
Quick Baby Cry Emotion Classifier - Uses feature extraction + simple ML
This is much faster to train on CPU than fine-tuning wav2vec2
"""
import os
import numpy as np
import librosa
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Configuration
EMOTION_LABELS = ['hunger', 'irritated', 'fear', 'normal', 'sad', 'happy']
SAMPLE_RATE = 16000
OUTPUT_DIR = "./baby_emotion_model_fast"

def extract_features(audio_path):
    """Extract audio features for ML classification"""
    try:
        audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        
        # Extract features
        features = []
        
        # MFCCs (most important for audio classification)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        features.extend(np.mean(mfccs, axis=1))
        features.extend(np.std(mfccs, axis=1))
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        features.append(np.mean(spectral_centroid))
        features.append(np.std(spectral_centroid))
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        features.append(np.mean(spectral_rolloff))
        
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)
        features.append(np.mean(zero_crossing_rate))
        
        # RMS energy
        rms = librosa.feature.rms(y=audio)
        features.append(np.mean(rms))
        features.append(np.std(rms))
        
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        features.append(float(tempo) if np.isscalar(tempo) else np.mean(tempo))
        
        return features
    except Exception as e:
        print(f"Error extracting features from {audio_path}: {e}")
        return None

def prepare_dataset():
    """Load dataset and extract features"""
    print("=" * 60)
    print("PREPARING DATASET")
    print("=" * 60)
    
    metadata_path = os.path.join("baby_cry_dataset", "metadata.csv")
    if not os.path.exists(metadata_path):
        print(f"❌ Metadata not found. Run prepare_dataset.py first!")
        return None, None
    
    df = pd.read_csv(metadata_path)
    
    print(f"\n📊 Dataset: {len(df)} files")
    print(df['emotion'].value_counts())
    
    # Extract features for all files
    print("\n🔍 Extracting features...")
    X = []
    y = []
    
    for idx, row in df.iterrows():
        features = extract_features(row['file_path'])
        if features is not None:
            X.append(features)
            y.append(EMOTION_LABELS.index(row['emotion']))
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(df)} files...")
    
    X = np.array(X)
    y = np.array(y)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n✓ Train: {len(X_train)}, Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test

def train_classifier():
    """Train Random Forest classifier"""
    print("\n" + "=" * 60)
    print("TRAINING BABY CRY CLASSIFIER")
    print("=" * 60)
    
    result = prepare_dataset()
    if result is None:
        return
    
    X_train, X_test, y_train, y_test = result
    
    # Train Random Forest
    print("\n🌲 Training Random Forest...")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    clf.fit(X_train, y_train)
    
    # Evaluate
    print("\n📊 Evaluation:")
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✓ Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=EMOTION_LABELS))
    
    # Save model
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    model_path = os.path.join(OUTPUT_DIR, "baby_cry_classifier.pkl")
    joblib.dump(clf, model_path)
    print(f"\n💾 Model saved to: {model_path}")
    
    # Save metadata
    metadata = {
        'emotion_labels': EMOTION_LABELS,
        'sample_rate': SAMPLE_RATE,
        'accuracy': accuracy,
        'model_type': 'RandomForest'
    }
    metadata_path = os.path.join(OUTPUT_DIR, "metadata.pkl")
    joblib.dump(metadata, metadata_path)
    print(f"✓ Metadata saved to: {metadata_path}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"✅ Model accuracy: {accuracy:.2%}")
    print(f"📁 Model location: {OUTPUT_DIR}")
    print("\nYou can now test it with: python test_model_fast.py <audio_file>")

if __name__ == "__main__":
    train_classifier()

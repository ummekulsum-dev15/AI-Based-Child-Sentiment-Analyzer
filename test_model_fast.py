"""
Fast Baby Cry Classifier Test Script
"""
import os
import numpy as np
import librosa
import joblib

MODEL_DIR = "./baby_emotion_model_fast"
SAMPLE_RATE = 16000

class BabyCryClassifierFast:
    """Fast baby cry classifier using Random Forest"""
    
    def __init__(self, model_dir=MODEL_DIR):
        if not os.path.exists(model_dir):
            raise Exception(f"Model not found at {model_dir}")
        
        # Load model
        self.model = joblib.load(os.path.join(model_dir, "baby_cry_classifier.pkl"))
        self.metadata = joblib.load(os.path.join(model_dir, "metadata.pkl"))
        self.emotion_labels = self.metadata['emotion_labels']
        
        # Map to app emotions
        self.emotion_mapping = {
            'hunger': 'hunger',
            'irritated': 'irritated',
            'fear': 'fear',
            'normal': 'normal',
            'sad': 'sad',
            'happy': 'normal'
        }
        
        print(f"✓ Fast baby cry model loaded (accuracy: {self.metadata['accuracy']:.2%})")
    
    def extract_features(self, audio_path=None, audio_array=None, sr=SAMPLE_RATE):
        """Extract audio features"""
        try:
            if audio_path:
                audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
            else:
                audio = audio_array
                if sr != SAMPLE_RATE:
                    audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE)
            
            features = []
            mfccs = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=13)
            features.extend(np.mean(mfccs, axis=1))
            features.extend(np.std(mfccs, axis=1))
            
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=SAMPLE_RATE)
            features.append(np.mean(spectral_centroid))
            features.append(np.std(spectral_centroid))
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=SAMPLE_RATE)
            features.append(np.mean(spectral_rolloff))
            
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)
            features.append(np.mean(zero_crossing_rate))
            
            rms = librosa.feature.rms(y=audio)
            features.append(np.mean(rms))
            features.append(np.std(rms))
            
            tempo, _ = librosa.beat.beat_track(y=audio, sr=SAMPLE_RATE)
            features.append(float(tempo) if np.isscalar(tempo) else np.mean(tempo))
            
            return features
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None
    
    def predict(self, audio_path):
        """Predict emotion from audio file"""
        features = self.extract_features(audio_path=audio_path)
        if features is None:
            return "❌ Error extracting features"
        
        features = np.array(features).reshape(1, -1)
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        
        predicted_emotion = self.emotion_labels[prediction]
        app_emotion = self.emotion_mapping.get(predicted_emotion, 'normal')
        
        result = f"🎵 Audio emotion: {app_emotion.capitalize()}"
        result += f"\n\n📊 Detailed predictions:"
        
        for emotion, prob in sorted(zip(self.emotion_labels, probabilities), key=lambda x: -x[1]):
            if prob > 0.05:
                result += f"\n  {emotion:15}: {prob:.2%}"
        
        return result
    
    def predict_from_array(self, audio_array, sample_rate):
        """Predict from numpy array"""
        features = self.extract_features(audio_array=audio_array, sr=sample_rate)
        if features is None:
            return "❌ Error extracting features"
        
        features = np.array(features).reshape(1, -1)
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        
        predicted_emotion = self.emotion_labels[prediction]
        app_emotion = self.emotion_mapping.get(predicted_emotion, 'normal')
        
        result = f"🎵 Audio emotion: {app_emotion.capitalize()}"
        result += f"\n\n📊 Detailed predictions:"
        
        for emotion, prob in sorted(zip(self.emotion_labels, probabilities), key=lambda x: -x[1]):
            if prob > 0.05:
                result += f"\n  {emotion:15}: {prob:.2%}"
        
        return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        classifier = BabyCryClassifierFast()
        path = sys.argv[1]
        if os.path.isfile(path):
            print(classifier.predict(path))
        else:
            print(f"❌ File not found: {path}")
    else:
        print("Usage: python test_model_fast.py <audio_file>")

"""
Baby Cry Audio Emotion Recognition - Test Script
Test the trained model on new audio files
"""
import os
import torch
import numpy as np
import librosa
import joblib
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

# Configuration
MODEL_DIR = "./baby_emotion_model"
SAMPLE_RATE = 16000
MAX_DURATION = 5  # seconds

class BabyCryClassifier:
    """Baby cry emotion classifier using trained wav2vec2 model"""
    
    def __init__(self, model_dir=MODEL_DIR):
        """Load model and processor"""
        self.model_dir = model_dir
        
        if not os.path.exists(model_dir):
            raise Exception(f"Model not found at {model_dir}. Please run train_model.py first.")
        
        print(f"🔄 Loading model from {model_dir}...")
        
        # Load feature extractor
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir)
        
        # Load model
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        
        # Load emotion labels
        emotion_labels_path = os.path.join(model_dir, "emotion_labels.pkl")
        self.emotion_labels = joblib.load(emotion_labels_path)
        
        # Map to baby emotion names for your app
        self.emotion_mapping = {
            'hunger': 'hunger',
            'irritated': 'irritated',
            'fear': 'fear',
            'normal': 'normal',
            'sad': 'sad',
            'happy': 'normal'  # Map happy to normal as per your requirements
        }
        
        print(f"✅ Model loaded successfully!")
        print(f"📋 Emotion labels: {self.emotion_labels}")
    
    def predict(self, audio_path):
        """Predict emotion from audio file"""
        if not os.path.exists(audio_path):
            return f"❌ File not found: {audio_path}"
        
        try:
            # Load and preprocess audio
            audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
            
            # Pad or trim
            max_samples = MAX_DURATION * SAMPLE_RATE
            if len(audio) > max_samples:
                audio = audio[:max_samples]
            else:
                audio = np.pad(audio, (0, max(max_samples - len(audio), 0)), mode='constant')
            
            # Process
            inputs = self.feature_extractor(
                audio,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)[0]
                predicted_class = torch.argmax(logits, dim=1).item()
            
            # Get results
            predicted_emotion = self.emotion_labels[predicted_class]
            confidence = probabilities[predicted_class].item()
            
            # Map to app emotion
            app_emotion = self.emotion_mapping.get(predicted_emotion, 'normal')
            
            # Build result
            result = f"🎵 Audio emotion: {app_emotion.capitalize()}"
            result += f"\n\n📊 Detailed predictions:"
            
            # Show all probabilities
            for i, (emotion, prob) in enumerate(zip(self.emotion_labels, probabilities)):
                if prob > 0.05:  # Only show emotions with >5% probability
                    result += f"\n  {emotion:15}: {prob:.2%}"
            
            return result
            
        except Exception as e:
            import traceback
            return f"❌ Error analyzing audio: {e}\n\n{traceback.format_exc()}"
    
    def predict_from_array(self, audio_array, sample_rate=SAMPLE_RATE):
        """Predict emotion from numpy array (for microphone input)"""
        try:
            # Resample if needed
            if sample_rate != SAMPLE_RATE:
                import librosa
                audio = librosa.resample(audio_array.astype(np.float32), 
                                        orig_sr=sample_rate, 
                                        target_sr=SAMPLE_RATE)
            else:
                audio = audio_array
            
            # Make mono if stereo
            if audio.ndim == 2:
                audio = np.mean(audio, axis=1)
            
            # Pad or trim
            max_samples = MAX_DURATION * SAMPLE_RATE
            if len(audio) > max_samples:
                audio = audio[:max_samples]
            else:
                audio = np.pad(audio, (0, max(max_samples - len(audio), 0)), mode='constant')
            
            # Process
            inputs = self.feature_extractor(
                audio,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)[0]
                predicted_class = torch.argmax(logits, dim=1).item()
            
            # Get results
            predicted_emotion = self.emotion_labels[predicted_class]
            confidence = probabilities[predicted_class].item()
            
            # Map to app emotion
            app_emotion = self.emotion_mapping.get(predicted_emotion, 'normal')
            
            result = f"🎵 Audio emotion: {app_emotion.capitalize()}"
            result += f"\n\n📊 Detailed predictions:"
            
            for i, (emotion, prob) in enumerate(zip(self.emotion_labels, probabilities)):
                if prob > 0.05:
                    result += f"\n  {emotion:15}: {prob:.2%}"
            
            return result
            
        except Exception as e:
            import traceback
            return f"❌ Error analyzing audio: {e}\n\n{traceback.format_exc()}"

def test_single_file(audio_path):
    """Test a single audio file"""
    classifier = BabyCryClassifier()
    result = classifier.predict(audio_path)
    print(f"\n{'=' * 60}")
    print(f"Testing: {audio_path}")
    print(f"{'=' * 60}")
    print(result)
    print(f"{'=' * 60}\n")

def test_folder(folder_path):
    """Test all audio files in a folder"""
    classifier = BabyCryClassifier()
    
    print(f"\n{'=' * 60}")
    print(f"TESTING ALL AUDIO FILES IN: {folder_path}")
    print(f"{'=' * 60}\n")
    
    audio_files = [f for f in os.listdir(folder_path) 
                   if f.lower().endswith(('.wav', '.mp3', '.ogg', '.flac'))]
    
    if not audio_files:
        print("❌ No audio files found!")
        return
    
    correct = 0
    total = 0
    
    for audio_file in audio_files:
        audio_path = os.path.join(folder_path, audio_file)
        # Try to infer true label from folder name
        parent_folder = os.path.basename(os.path.dirname(audio_path))
        true_label = parent_folder.lower()
        
        result = classifier.predict(audio_path)
        predicted = result.split("Audio emotion: ")[1].split("\n")[0] if "Audio emotion:" in result else "unknown"
        
        print(f"📁 {audio_file:30} | True: {true_label:15} | Predicted: {predicted}")
        
        total += 1
        if true_label in predicted.lower():
            correct += 1
    
    if total > 0:
        accuracy = correct / total
        print(f"\n{'=' * 60}")
        print(f"📊 Accuracy on this folder: {accuracy:.2%} ({correct}/{total})")
        print(f"{'=' * 60}\n")

def main():
    """Main testing function"""
    print("\n" + "🎵" * 30)
    print("BABY CRY EMOTION RECOGNITION - TESTING")
    print("🎵" * 30 + "\n")
    
    # Test single file
    import sys
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        if os.path.exists(audio_path):
            if os.path.isfile(audio_path):
                test_single_file(audio_path)
            elif os.path.isdir(audio_path):
                test_folder(audio_path)
        else:
            print(f"❌ Path not found: {audio_path}")
    else:
        print("Usage:")
        print("  python test_model.py <audio_file_or_folder>")
        print("\nExample:")
        print("  python test_model.py baby_cry_dataset/hunger/cry_001.wav")
        print("  python test_model.py baby_cry_dataset/hunger/")
        
        # Demo test
        print("\n" + "=" * 60)
        print("Running demo test...")
        print("=" * 60)
        
        dataset_path = os.path.join(os.path.dirname(__file__), "baby_cry_dataset")
        if os.path.exists(dataset_path):
            # Test one file from each emotion
            for emotion in os.listdir(dataset_path):
                emotion_folder = os.path.join(dataset_path, emotion)
                if os.path.isdir(emotion_folder):
                    files = [f for f in os.listdir(emotion_folder) 
                            if f.lower().endswith(('.wav', '.mp3', '.ogg', '.flac'))]
                    if files:
                        test_single_file(os.path.join(emotion_folder, files[0]))

if __name__ == "__main__":
    main()

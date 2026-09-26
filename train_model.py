"""
Baby Cry Audio Emotion Recognition Model Training Script
Fine-tunes wav2vec2 model on baby cry dataset for emotion recognition
"""
import os
import torch
import numpy as np
import librosa
from pathlib import Path
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset, Audio, Features, ClassLabel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import joblib

# Configuration
EMOTION_LABELS = ['hunger', 'irritated', 'fear', 'normal', 'sad', 'happy']
MODEL_NAME = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
SAMPLE_RATE = 16000
MAX_DURATION = 5  # seconds
OUTPUT_DIR = "./baby_emotion_model"

def load_audio_file(file_path):
    """Load and preprocess audio file"""
    try:
        # Load audio at target sample rate
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        
        # Trim or pad to max duration
        max_samples = MAX_DURATION * SAMPLE_RATE
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        else:
            audio = np.pad(audio, (0, max(max_samples - len(audio), 0)), mode='constant')
        
        return audio
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def prepare_dataset(dataset_path):
    """Load and prepare dataset for training"""
    print("=" * 60)
    print("PREPARING DATASET FOR TRAINING")
    print("=" * 60)
    
    # Check if metadata.csv exists
    metadata_path = os.path.join(dataset_path, "metadata.csv")
    if not os.path.exists(metadata_path):
        print(f"❌ Metadata file not found at: {metadata_path}")
        print("Please run prepare_dataset.py first!")
        return None
    
    # Read metadata
    import pandas as pd
    df = pd.read_csv(metadata_path)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  Total files: {len(df)}")
    print(f"  Emotions: {df['emotion'].value_counts().to_dict()}")
    
    # Filter out invalid files
    valid_files = []
    valid_emotions = []
    
    print("\n🔍 Loading and validating audio files...")
    for idx, row in df.iterrows():
        file_path = row['file_path']
        emotion = row['emotion']
        
        if os.path.exists(file_path):
            valid_files.append(file_path)
            valid_emotions.append(emotion)
        else:
            print(f"  ⚠️  File not found: {file_path}")
    
    if len(valid_files) == 0:
        print("❌ No valid audio files found!")
        return None
    
    print(f"\n✅ Valid files: {len(valid_files)}")
    
    # Create DataFrame
    df_valid = pd.DataFrame({'file_path': valid_files, 'emotion': valid_emotions})
    
    # Split into train and test
    train_df, test_df = train_test_split(
        df_valid, 
        test_size=0.2, 
        stratify=df_valid['emotion'],
        random_state=42
    )
    
    # For CPU training, use a smaller subset for faster training
    max_train_samples = 150  # Very small subset for VERY fast CPU training  
    if len(train_df) > max_train_samples:
        print(f"\n⚡ Using subset of {max_train_samples} samples for fast CPU training...")
        train_df = train_df.sample(n=max_train_samples, random_state=42).reset_index(drop=True)
    
    print(f"\n📈 Train/Test Split:")
    print(f"  Training samples: {len(train_df)}")
    print(f"  Testing samples: {len(test_df)}")
    
    # Create Hugging Face Datasets
    train_dataset = create_hf_dataset(train_df)
    test_dataset = create_hf_dataset(test_df)
    
    return train_dataset, test_dataset

def create_hf_dataset(df):
    """Create Hugging Face Dataset from DataFrame"""
    # Load all audio files
    audio_data = []
    labels = []
    
    print("  Loading audio files into memory...")
    for idx, row in df.iterrows():
        audio = load_audio_file(row['file_path'])
        if audio is not None:
            # Store as dict instead of Audio feature to avoid torchcodec dependency
            audio_data.append({'input_values': audio})
            labels.append(EMOTION_LABELS.index(row['emotion']))
    
    # Create simple dataset without Audio feature (avoid torchcodec dependency)
    dataset_dict = {
        'input_values': [d['input_values'] for d in audio_data],
        'label': labels
    }
    
    return Dataset.from_dict(dataset_dict)

def compute_metrics(eval_pred):
    """Compute accuracy and F1 score"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1
    }

def train_model(train_dataset, test_dataset):
    """Train the wav2vec2 model on baby cry dataset"""
    print("\n" + "=" * 60)
    print("TRAINING BABY CRY EMOTION RECOGNITION MODEL")
    print("=" * 60)
    
    # Load feature extractor
    print(f"\n🔄 Loading feature extractor from: {MODEL_NAME}")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    
    # Preprocess function
    def preprocess_function(examples):
        # Audio already loaded as numpy arrays
        audio_arrays = examples['input_values']
        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=SAMPLE_RATE,
            truncation=True,
            padding=True,
            max_length=16000 * MAX_DURATION,
        )
        inputs['labels'] = examples['label']
        return inputs
    
    # Process datasets
    print("\n🔧 Processing datasets...")
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=['input_values']
    )
    
    test_dataset = test_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=['input_values']
    )
    
    # Load model
    print("\n🤖 Loading model...")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL_NAME,
        id2label={i: label for i, label in enumerate(EMOTION_LABELS)},
        label2id={label: i for i, label in enumerate(EMOTION_LABELS)},
        ignore_mismatched_sizes=True
    )
    
    # Update config for correct number of labels
    model.config.num_labels = len(EMOTION_LABELS)
    model.config.problem_type = "single_label_classification"
    model.config.id2label = {i: label for i, label in enumerate(EMOTION_LABELS)}
    model.config.label2id = {label: i for i, label in enumerate(EMOTION_LABELS)}
    
    # Freeze feature extractor
    for param in model.wav2vec2.parameters():
        param.requires_grad = False
    
    # Training arguments - optimized for VERY fast CPU training
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,  # Minimal epochs for fast training
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=3e-5,
        warmup_steps=20,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        fp16=False,
        report_to="none",
        dataloader_pin_memory=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )
    
    # Train
    print("\n🚀 Starting training...")
    trainer.train()
    
    # Evaluate
    print("\n📊 Evaluating model...")
    eval_results = trainer.evaluate()
    print(f"\n✅ Final Results:")
    print(f"  Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"  F1 Score: {eval_results['eval_f1']:.4f}")
    
    # Save model
    print(f"\n💾 Saving model to: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    feature_extractor.save_pretrained(OUTPUT_DIR)
    
    # Save emotion labels
    emotion_mapping_path = os.path.join(OUTPUT_DIR, "emotion_labels.pkl")
    joblib.dump(EMOTION_LABELS, emotion_mapping_path)
    print(f"  ✓ Emotion labels saved to: {emotion_mapping_path}")
    
    return trainer, eval_results

def main():
    """Main training pipeline"""
    print("\n" + "🎵" * 30)
    print("BABY CRY EMOTION RECOGNITION - TRAINING PIPELINE")
    print("🎵" * 30 + "\n")
    
    # Find dataset
    dataset_path = os.path.join(os.path.dirname(__file__), "baby_cry_dataset")
    if not os.path.exists(dataset_path):
        print("❌ Dataset not found!")
        print("Please run prepare_dataset.py first to download and organize the dataset.")
        return
    
    # Prepare dataset
    result = prepare_dataset(dataset_path)
    if result is None:
        return
    
    train_dataset, test_dataset = result
    
    # Train model
    trainer, eval_results = train_model(train_dataset, test_dataset)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\n✅ Model saved to: {OUTPUT_DIR}")
    print(f"📊 Final Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"📊 Final F1 Score: {eval_results['eval_f1']:.4f}")
    print("\nNext steps:")
    print("  1. Run: python test_model.py (to test on new audio)")
    print("  2. Update main.py to use the new model")
    print("=" * 60)

if __name__ == "__main__":
    main()

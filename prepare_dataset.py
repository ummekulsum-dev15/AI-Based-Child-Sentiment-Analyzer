"""
Baby Cry Audio Dataset Preprocessing Script
Prepares audio files for training by converting to WAV format and standardizing sample rate
"""
import os
import kagglehub
import shutil
from pathlib import Path

# Mapping: folder name -> emotion label
EMOTION_MAPPING = {
    'hungry': 'hunger',
    'discomfort': 'irritated',
    'scared': 'fear',
    'silence': 'normal',
    'tired': 'sad',
    'laugh': 'happy',
    'laughing': 'happy'
}

def download_and_organize_dataset():
    """Download dataset from Kaggle and organize it with proper structure"""
    print("=" * 60)
    print("DOWNLOADING & ORGANIZING BABY CRY DATASET")
    print("=" * 60)
    
    # Download from Kaggle
    print("\n📥 Downloading dataset from Kaggle...")
    source_path = kagglehub.dataset_download("mennaahmed23/baby-cry")
    print(f"✓ Dataset downloaded to: {source_path}")
    
    # Create organized dataset folder
    target_path = os.path.join(os.path.dirname(__file__), "baby_cry_dataset")
    os.makedirs(target_path, exist_ok=True)
    
    print(f"\n📂 Organizing dataset in: {target_path}")
    
    # Copy and organize files
    total_files = 0
    
    # Check if there's a parent "cry" folder
    if os.path.exists(os.path.join(source_path, "cry")):
        source_path = os.path.join(source_path, "cry")
        print(f"Found 'cry' parent folder, using: {source_path}")
    
    for emotion_folder in os.listdir(source_path):
        source_folder = os.path.join(source_path, emotion_folder)
        
        if not os.path.isdir(source_folder):
            continue
        
        # Map folder name to emotion
        emotion_lower = emotion_folder.lower()
        if emotion_lower in EMOTION_MAPPING:
            emotion_label = EMOTION_MAPPING[emotion_lower]
            target_folder = os.path.join(target_path, emotion_label)
            os.makedirs(target_folder, exist_ok=True)
            
            # Copy audio files
            file_count = 0
            for file in os.listdir(source_folder):
                if file.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
                    src_file = os.path.join(source_folder, file)
                    dst_file = os.path.join(target_folder, file)
                    shutil.copy2(src_file, dst_file)
                    file_count += 1
                    total_files += 1
            
            print(f"  ✓ {emotion_folder:15} -> {emotion_label:15} ({file_count} files)")
    
    print(f"\n✅ Total files organized: {total_files}")
    print(f"📁 Dataset saved to: {target_path}")
    
    # Create metadata file
    create_metadata(target_path)
    
    return target_path

def create_metadata(dataset_path):
    """Create a metadata CSV file for training"""
    metadata_path = os.path.join(dataset_path, "metadata.csv")
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        f.write("file_path,emotion\n")
        
        for emotion in os.listdir(dataset_path):
            emotion_folder = os.path.join(dataset_path, emotion)
            if not os.path.isdir(emotion_folder):
                continue
            
            for file in os.listdir(emotion_folder):
                if file.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
                    file_path = os.path.join(emotion_folder, file)
                    f.write(f"{file_path},{emotion}\n")
    
    print(f"📝 Metadata saved to: {metadata_path}")

if __name__ == "__main__":
    dataset_path = download_and_organize_dataset()
    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETE!")
    print("=" * 60)
    print(f"\nYou can now use this dataset to train the model.")
    print(f"Run: python train_model.py")

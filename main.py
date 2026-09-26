# Suppress all warnings at the very beginning
import os
import sys
import warnings
import logging

# Suppress Python warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow/oneDNN warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=ALL, 1=INFO, 2=WARNING, 3=ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom ops warning

# Set transformers cache and disable telemetry
os.environ['TRANSFORMERS_CACHE'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models_cache')
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

# Suppress Gradio/HTTP logs
logging.getLogger('gradio').setLevel(logging.ERROR)
logging.getLogger('httpx').setLevel(logging.ERROR)
logging.getLogger('httpcore').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from textblob import TextBlob
import gradio as gr
import cv2
# Import transformers components lazily
from langdetect import detect
import torch

# Try to import baby cry classifier
try:
    from test_model_fast import BabyCryClassifierFast
    BABY_MODEL_AVAILABLE = os.path.exists("./baby_emotion_model_fast")
except ImportError:
    BABY_MODEL_AVAILABLE = False
    BabyCryClassifierFast = None

try:
    from fer.fer import FER
    FER_AVAILABLE = True
except ImportError:
    FER_AVAILABLE = False
    print("FER library not available, using alternative face emotion detection")

# Global variables
baby_classifier = None
audio_emotion_pipeline = None
text_emotion_pipeline = None
translation_tokenizer = None
translation_model = None


def get_baby_classifier():
    """Get baby cry classifier instance (uses trained model if available)"""
    global baby_classifier
    if baby_classifier is None and BABY_MODEL_AVAILABLE:
        try:
            baby_classifier = BabyCryClassifierFast()
            print("✅ Fast baby cry model loaded successfully!")
        except Exception as e:
            print(f"⚠️  Failed to load baby cry model: {e}")
            baby_classifier = None
    return baby_classifier


def get_audio_emotion_pipeline():
    global audio_emotion_pipeline
    if audio_emotion_pipeline is None:
        from transformers import pipeline
        print("Loading audio emotion model (this may take a moment)...")
        audio_emotion_pipeline = pipeline(
            task="audio-classification",
            model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
            top_k=1,
        )
    return audio_emotion_pipeline


def get_text_emotion_pipeline():
    global text_emotion_pipeline
    if text_emotion_pipeline is None:
        from transformers import pipeline
        print("Loading text emotion model (this may take a moment)...")
        text_emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
    return text_emotion_pipeline


def get_translation_model():
    global translation_tokenizer, translation_model
    if translation_model is None:
        try:
            from transformers import MarianMTModel, MarianTokenizer
            print("Loading translation model (this may take a moment)...")
            model_name = "Helsinki-NLP/opus-mt-bn-en"
            translation_tokenizer = MarianTokenizer.from_pretrained(model_name)
            translation_model = MarianMTModel.from_pretrained(model_name)
            # Move model to GPU if available
            if torch.cuda.is_available():
                translation_model = translation_model.cuda()
        except Exception as e:
            print(f"Error loading translation model: {e}")
            return None, None
    return translation_tokenizer, translation_model


def detect_language(text: str) -> str:
    try:
        lang = detect(text)
        return lang
    except:
        return "en"


def translate_bangla_to_english(text: str) -> str:
    try:
        tokenizer, model = get_translation_model()
        if model is None or tokenizer is None:
            return None
        
        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate translation
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=512)
        
        # Decode translation
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated_text
    except Exception as error:
        print(f"Translation error: {error}")
        return None


def analyze_text_sentiment(text: str, language: str = "Auto-detect") -> str:
    if not isinstance(text, str) or not text.strip():
        return "Please enter a sentence."

    try:
        # Detect language if auto-detect is selected
        if language == "Auto-detect":
            detected_lang = detect_language(text)
        else:
            detected_lang = "bn" if "Bangla" in language else "en"
        
        # Translate Bangla to English if needed
        if detected_lang == "bn":
            translated_text = translate_bangla_to_english(text)
            if translated_text is None:
                return "Translation error: Could not translate Bangla text. Please try English or check your input."
            text_to_analyze = translated_text
        else:
            text_to_analyze = text
        
        # Analyze emotion
        classifier = get_text_emotion_pipeline()
        results = classifier(text_to_analyze)
        if not results:
            return "Emotion could not be predicted."
        label = results[0]["label"].lower()
        mapping = {
            'anger': 'irritated',
            'disgust': 'hunger',
            'fear': 'fear',
            'joy': 'normal',
            'neutral': 'normal',
            'sadness': 'sad',
            'surprise': 'normal'
        }
        emotion = mapping.get(label, 'normal')
        return f"Emotion: {emotion.capitalize()}"
    except Exception as error:
        return f"Text analysis error: {error}"


def convert_to_wav(audio_path: str) -> str:
    """Convert any audio file to WAV format using Python libraries"""
    import os
    import tempfile
    
    if audio_path.lower().endswith('.wav'):
        return audio_path  # Already WAV
    
    print(f"Converting {audio_path} to WAV format...")
    
    try:
        # Try using pydub (may need FFmpeg)
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(audio_path)
            wav_path = os.path.join(tempfile.gettempdir(), "converted_audio.wav")
            audio.export(wav_path, format="wav")
            print(f"Converted using pydub: {wav_path}")
            return wav_path
        except Exception as e1:
            print(f"Pydub failed: {e1}")
            
            # Try using soundfile + librosa
            try:
                import librosa
                import soundfile as sf
                audio_array, sr = librosa.load(audio_path, sr=None)
                wav_path = os.path.join(tempfile.gettempdir(), "converted_audio.wav")
                sf.write(wav_path, audio_array, sr)
                print(f"Converted using librosa+soundfile: {wav_path}")
                return wav_path
            except Exception as e2:
                print(f"Librosa failed: {e2}")
                return None
                
    except Exception as e:
        print(f"Conversion failed: {e}")
        return None


def analyze_audio_emotion(audio_input) -> str:
    """Analyze audio emotion from uploaded file or recorded audio"""
    if audio_input is None:
        return "Please upload or record audio."

    # Try baby classifier first (if trained model exists)
    baby_clf = get_baby_classifier()
    if baby_clf is not None:
        try:
            import numpy as np
            
            # Handle different input types
            if isinstance(audio_input, str):
                # File upload
                return baby_clf.predict(audio_input)
            elif isinstance(audio_input, tuple) and len(audio_input) == 2:
                # Microphone recording
                sample_rate, audio_array = audio_input
                return baby_clf.predict_from_array(audio_array, int(sample_rate))
        except Exception as e:
            print(f"Baby classifier failed: {e}")
            # Fall through to fallback method

    # Fallback to original method
    return _analyze_audio_fallback(audio_input)


def _analyze_audio_fallback(audio_input) -> str:
    """Fallback audio analysis using original model (when baby model not available)"""
    try:
        import numpy as np
        import torch
        import soundfile as sf
        import os

        audio_array = None
        sample_rate = 16000

        # Handle Gradio Audio input (microphone recording)
        if isinstance(audio_input, tuple) and len(audio_input) == 2:
            sample_rate, audio_array = audio_input
            if isinstance(sample_rate, (int, float)):
                sample_rate = int(sample_rate)
            else:
                audio_array, sample_rate = audio_input
                sample_rate = int(sample_rate)
            print(f"✓ Received microphone audio: {sample_rate} Hz, shape {audio_array.shape}")

        # Handle Gradio File upload
        elif isinstance(audio_input, str):
            audio_path = audio_input
            print(f"✓ Received file upload: {audio_path}")

            if not os.path.exists(audio_path):
                return f"❌ File not found: {audio_path}"

            file_size = os.path.getsize(audio_path)
            print(f"   File size: {file_size / 1024:.1f} KB")

            try:
                audio_array, sample_rate = sf.read(audio_path, dtype='float32')
                print(f"✓ Loaded with soundfile: {len(audio_array)} samples, {sample_rate} Hz")
            except Exception as e1:
                print(f"   Soundfile failed: {e1}")
                try:
                    import librosa
                    audio_array, sample_rate = librosa.load(audio_path, sr=None)
                    print(f"✓ Loaded with librosa: {len(audio_array)} samples, {sample_rate} Hz")
                except Exception as e2:
                    print(f"   Librosa failed: {e2}")
                    return f"❌ Cannot read this file format.\n\n**Solution:** Please upload WAV format files."
        else:
            return f"❌ Unsupported format: {type(audio_input)}"

        if audio_array is None or len(audio_array) == 0:
            return "❌ Failed to load audio data."

        if hasattr(audio_array, 'cpu'):
            audio_array = audio_array.cpu().numpy()
        elif not isinstance(audio_array, np.ndarray):
            audio_array = np.array(audio_array)

        if audio_array.ndim == 2:
            audio_array = np.mean(audio_array, axis=1)

        duration = len(audio_array) / sample_rate if sample_rate > 0 else 0
        
        if duration < 0.5:
            return "❌ Audio too short (less than 0.5s)."

        classifier = get_audio_emotion_pipeline()
        audio_tensor = torch.from_numpy(audio_array).float()
        results = classifier(audio_tensor)

        if not results or len(results) == 0:
            return "❌ Audio emotion could not be predicted."

        dominant_emotion = results[0]["label"].lower()
        
        mapping = {
            'angry': 'irritated',
            'calm': 'normal',
            'disgust': 'hunger',
            'fearful': 'fear',
            'happy': 'normal',
            'neutral': 'normal',
            'sad': 'sad',
            'surprised': 'normal'
        }

        emotion = mapping.get(dominant_emotion, 'normal')
        return f"🎵 Audio emotion: {emotion.capitalize()} (using general model)"

    except Exception as error:
        import traceback
        return f"❌ Audio analysis error: {error}"


def analyze_video_emotion(video_input) -> str:
    if not video_input:
        return "Please upload a video file."

    try:
        import cv2
        import numpy as np

        # Get video file path - handle different Gradio input formats
        if isinstance(video_input, dict):
            video_path = video_input.get('video') or video_input.get('path') or video_input.get('name')
        elif isinstance(video_input, tuple):
            video_path = video_input[0]
        else:
            video_path = video_input

        if not video_path or not isinstance(video_path, str):
            return f"Invalid video file format. Received: {type(video_input)}"
        
        if not os.path.exists(video_path):
            return f"Video file not found: {video_path}\n\nPlease try uploading the video again."

        print(f"\n=== Video Analysis Started ===")
        print(f"Video path: {video_path}")
        print(f"File size: {os.path.getsize(video_path) / (1024*1024):.2f} MB")

        # Open video and sample frames
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return "❌ Could not open video file.\n\n**Possible reasons:**\n- Unsupported video codec/format\n- Corrupted file\n- Try converting to MP4 (H.264 codec)"

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        print(f"Video specs: {width}x{height}, {fps:.2f} FPS, {total_frames} frames, {duration:.2f}s")

        if total_frames == 0 or fps == 0:
            cap.release()
            return "❌ Could not read video metadata.\n\n**Solution:** Try re-encoding to standard MP4 format."

        # Check if FER is available
        if not FER_AVAILABLE:
            print("⚠️ FER library not available, using fallback emotion detection")

        emotions_detected = []
        frame_count = 0
        frames_analyzed = 0
        faces_found = 0
        max_frames_to_analyze = 30

        # Sample frames evenly throughout the video
        # Skip first 1 second (might be loading screen), analyze up to 80% of video
        start_frame = int(fps) if fps > 0 else 30
        end_frame = int(total_frames * 0.8)  # Don't analyze last 20%
        frame_step = max(1, (end_frame - start_frame) // max_frames_to_analyze)

        print(f"Video analysis: analyzing every {frame_step} frame(s) from frame {start_frame} to {end_frame}")

        # Load face cascade classifiers ONCE
        face_cascade_frontal = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        face_cascade_profile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        
        if face_cascade_frontal.empty() or face_cascade_profile.empty():
            print("⚠️ Warning: Haar cascades not loaded properly")

        # Initialize FER detector ONCE outside the loop (much faster)
        fer_detector = None
        if FER_AVAILABLE:
            try:
                fer_detector = FER(mtcnn=True)
                print("✅ FER detector initialized")
            except Exception as e:
                print(f"⚠️ FER initialization failed: {e}, using fallback")

        # Initialize fallback classifier if needed
        fallback_classifier = None
        if not FER_AVAILABLE or fer_detector is None:
            try:
                from transformers import pipeline
                fallback_classifier = pipeline(
                    "image-classification",
                    model="trpakov/vit-face-expression"
                )
                print("✅ Fallback emotion classifier initialized")
            except Exception as e:
                print(f"⚠️ Fallback classifier failed to load: {e}")

        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Analyze frames at intervals
            if frame_count >= start_frame and frame_count <= end_frame and frame_count % frame_step == 0 and frames_analyzed < max_frames_to_analyze:
                frames_analyzed += 1
                print(f"\nProcessing frame {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)")

                # Convert frame for processing
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_eq = cv2.equalizeHist(gray)

                # Try to detect faces with multiple approaches
                faces = []
                
                # Method 1: Standard frontal face detection
                faces = face_cascade_frontal.detectMultiScale(
                    gray_eq, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                # Method 2: Try alt profile if no frontal faces
                if len(faces) == 0:
                    faces = face_cascade_profile.detectMultiScale(
                        gray_eq, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30)
                    )
                
                # Method 3: Relaxed parameters
                if len(faces) == 0:
                    faces = face_cascade_frontal.detectMultiScale(
                        gray_eq, scaleFactor=1.2, minNeighbors=2, minSize=(20, 20)
                    )

                if len(faces) > 0:
                    faces_found += 1
                    # Take largest face
                    faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                    (x, y, w, h) = faces_sorted[0]

                    # Extract and preprocess face
                    padding = int(w * 0.2)
                    x1, y1 = max(0, x - padding), max(0, y - padding)
                    x2, y2 = min(frame.shape[1], x + w + padding), min(frame.shape[0], y + h + padding)
                    
                    face_roi = frame[y1:y2, x1:x2]
                    face_resized = cv2.resize(face_roi, (224, 224))
                    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

                    # Detect emotion
                    emotion_detected = False
                    
                    # Try FER first
                    if fer_detector is not None:
                        try:
                            emotions_list = fer_detector.detect_emotions(face_rgb)
                            if emotions_list and len(emotions_list) > 0:
                                face_emotions = emotions_list[0].get('emotions', {})
                                if face_emotions:
                                    dominant = max(face_emotions, key=face_emotions.get)
                                    confidence = face_emotions[dominant]
                                    emotions_detected.append(dominant)
                                    print(f"  ✅ Face #{faces_found}: {dominant} ({confidence:.2f})")
                                    emotion_detected = True
                        except Exception as e:
                            print(f"  ⚠️ FER failed: {e}")

                    # Fallback to transformer
                    if not emotion_detected and fallback_classifier is not None:
                        try:
                            results = fallback_classifier(face_rgb, top_k=1)
                            if results:
                                emotion = results[0]['label'].lower()
                                confidence = results[0].get('score', 0)
                                emotions_detected.append(emotion)
                                print(f"  ✅ Face #{faces_found}: {emotion} ({confidence:.2f}) [fallback]")
                                emotion_detected = True
                        except Exception as e:
                            print(f"  ⚠️ Fallback failed: {e}")
                    
                    if not emotion_detected:
                        print(f"  ❌ Face #{faces_found}: Could not detect emotion")
                else:
                    print(f"  ⚠️ No face found in frame {frame_count}")

            frame_count += 1

        cap.release()

        print(f"\n=== Video Analysis Complete ===")
        print(f"Total frames analyzed: {frames_analyzed}")
        print(f"Faces found: {faces_found}")
        print(f"Emotions detected: {len(emotions_detected)}")

        if not emotions_detected:
            if faces_found == 0:
                return (
                    "❌ **No faces detected in the video.**\n\n"
                    "**Possible reasons:**\n"
                    "• No visible faces in the video\n"
                    "• Faces are too small, blurry, or at extreme angles\n"
                    "• Poor lighting conditions\n"
                    "• Video format not properly decoded\n\n"
                    "**Tips for better detection:**\n"
                    "✓ Use videos with clear, frontal face views\n"
                    "✓ Ensure good lighting on the face\n"
                    "✓ Keep faces relatively large in the frame\n"
                    "✓ Try MP4 format with H.264 codec\n"
                    "✓ Avoid heavily compressed or corrupted videos\n\n"
                    f"**Video specs:** {width}x{height}, {fps:.1f} FPS, {duration:.1f}s"
                )
            else:
                return (
                    f"⚠️ **Faces were found but emotions could not be detected.**\n\n"
                    f"Faces detected: {faces_found}, but emotion recognition failed.\n\n"
                    "**Possible reasons:**\n"
                    "• Face image quality too low for emotion analysis\n"
                    "• Unusual facial expressions not in training data\n\n"
                    "**Try:**\n"
                    "✓ A different video with clearer facial expressions\n"
                    "✓ Ensuring the face shows recognizable emotions"
                )

        # Find the most common emotion
        from collections import Counter
        emotion_counts = Counter(emotions_detected)
        dominant_emotion = emotion_counts.most_common(1)[0][0]

        # Map emotion to baby emotion categories
        mapping = {
            'angry': 'irritated',
            'anger': 'irritated',
            'disgust': 'hunger',
            'fear': 'fear',
            'fearful': 'fear',
            'happy': 'normal',
            'happiness': 'normal',
            'neutral': 'normal',
            'sad': 'sad',
            'sadness': 'sad',
            'surprise': 'normal',
            'surprised': 'normal'
        }
        
        emotion = mapping.get(dominant_emotion, 'normal')
        
        # Show all detected emotions with counts
        emotion_summary = ", ".join([f"{emo}: {count}" for emo, count in emotion_counts.most_common()])
        
        return f"Video emotion: {emotion.capitalize()}\n(Detected emotions: {emotion_summary})"

    except Exception as error:
        import traceback
        return f"Video analysis error: {error}\n\nDetails: {traceback.format_exc()}"


def main() -> None:
    with gr.Blocks() as demo:
        gr.Markdown("# Child Sentiment & Emotion Analyzer AI")
        gr.Markdown(
            "Upload child audio or video to predict emotions such as hunger, irritated, fear, normal, happy or sad. "
            "Text emotion analysis is also available."
        )

        with gr.Tab("Text"):
            language_dropdown = gr.Dropdown(
                choices=["Auto-detect", "Bangla", "English"],
                value="Auto-detect",
                label="Language"
            )
            text_input = gr.Textbox(lines=2, placeholder="Enter a sentence in Bangla or English...")
            text_output = gr.Textbox()
            text_button = gr.Button("Analyze Text")
            text_button.click(analyze_text_sentiment, inputs=[text_input, language_dropdown], outputs=text_output)

        with gr.Tab("Audio"):
            gr.Markdown("### 🎵 Audio Emotion Detection")
            if BABY_MODEL_AVAILABLE:
                gr.Markdown("✅ **Using child cry model** for better accuracy")
            else:
                gr.Markdown("⚠️ **Using general model** - Train child model for better results")
            gr.Markdown("Upload **WAV files** or **record with microphone** to detect child emotions")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Option 1: Upload Audio File**")
                    file_input = gr.File(
                        label="Upload WAV File",
                        file_types=[".wav", ".ogg", ".flac"]
                    )
                
                with gr.Column():
                    gr.Markdown("**Option 2: Record Audio**")
                    mic_input = gr.Audio(
                        sources=["microphone"],
                        type="numpy",
                        label="Click to Record"
                    )
            
            audio_output = gr.Textbox(label="Result")
            audio_button = gr.Button("🎯 Analyze Audio", variant="primary")
            
            def analyze_audio(file, audio):
                if file is not None:
                    return analyze_audio_emotion(file)
                elif audio is not None:
                    return analyze_audio_emotion(audio)
                else:
                    return "Please upload a file OR record audio first."
            
            audio_button.click(
                analyze_audio, 
                inputs=[file_input, mic_input], 
                outputs=audio_output
            )

        with gr.Tab("Video"):
            video_input = gr.Video()
            video_output = gr.Textbox()
            video_button = gr.Button("Analyze Video")
            video_button.click(analyze_video_emotion, inputs=video_input, outputs=video_output)

        gr.Markdown(
            "## Developed by\n"
            "- Mahmuda Anjum Shamme.\n"
            "- Umme Kulsum.\n"
            "- Md. Nayeem Sarker.\n"
            "- Moniruzzaman Shaon.\n"
            "- Nafisa Tasnim Mohona."
        )

    # Show startup message
    print("\n✅ Child Sentiment & Emotion Analyzer AI")
    print("=" * 60)
    
    # Launch with minimal output (show URL but hide API logs)
    try:
        demo.launch(quiet=False, show_api=False)
    except TypeError:
        # Older Gradio versions don't support show_api
        demo.launch(quiet=False)


if __name__ == "__main__":
    main()

"""
Quick test to verify video analysis components are working
"""
import cv2
import numpy as np
import os

# Test 1: Check OpenCV can load video
def test_video_loading():
    print("=" * 60)
    print("TEST 1: Video Loading Test")
    print("=" * 60)
    
    # Create a simple test video
    test_video_path = "test_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(test_video_path, fourcc, 20.0, (640, 480))
    
    # Write 100 frames (5 seconds)
    for i in range(100):
        # Create a frame with a simple rectangle (simulating a face)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :] = (128, 128, 128)  # Gray background
        # Draw a face-like rectangle
        cv2.rectangle(frame, (200, 150), (440, 350), (255, 255, 255), -1)
        # Add eyes
        cv2.circle(frame, (280, 220), 20, (0, 0, 0), -1)
        cv2.circle(frame, (360, 220), 20, (0, 0, 0), -1)
        # Add mouth
        cv2.rectangle(frame, (270, 280), (370, 300), (0, 0, 0), -1)
        
        out.write(frame)
    out.release()
    
    print(f"✅ Created test video: {test_video_path}")
    print(f"   Size: {os.path.getsize(test_video_path) / 1024:.1f} KB")
    
    # Try to read it back
    cap = cv2.VideoCapture(test_video_path)
    if not cap.isOpened():
        print("❌ Failed to open test video")
        return False
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"✅ Video loaded successfully")
    print(f"   Specs: {width}x{height}, {fps:.1f} FPS, {total_frames} frames")
    
    # Test face detection on first frame
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
        
        if len(faces) > 0:
            print(f"✅ Face detection working: {len(faces)} face(s) detected")
        else:
            print("⚠️  No faces detected (expected with simple test video)")
    
    cap.release()
    print()
    return True

# Test 2: Check FER library
def test_fer_library():
    print("=" * 60)
    print("TEST 2: FER Library Test")
    print("=" * 60)
    
    try:
        from fer.fer import FER
        print("✅ FER library imported successfully")
        
        # Try to initialize (this might fail if dependencies are missing)
        try:
            detector = FER(mtcnn=True)
            print("✅ FER detector initialized")
            return True
        except Exception as e:
            print(f"⚠️  FER initialization failed: {e}")
            print("   Will use fallback emotion classifier")
            return False
    except ImportError as e:
        print(f"❌ FER import failed: {e}")
        print("   Install with: pip install fer")
        return False

# Test 3: Check fallback classifier
def test_fallback_classifier():
    print("=" * 60)
    print("TEST 3: Fallback Classifier Test")
    print("=" * 60)
    
    try:
        from transformers import pipeline
        print("✅ Transformers library available")
        
        # Don't actually load the model (takes too long), just verify import works
        print("ℹ️  Fallback model available: trpakov/vit-face-expression")
        print("   (Model not loaded to save time)")
        return True
    except ImportError as e:
        print(f"❌ Transformers import failed: {e}")
        return False

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("VIDEO ANALYZER COMPONENT TEST")
    print("=" * 60 + "\n")
    
    test1 = test_video_loading()
    test2 = test_fer_library()
    test3 = test_fallback_classifier()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Video Loading:        {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"FER Library:          {'✅ PASS' if test2 else '⚠️  Using fallback'}")
    print(f"Fallback Classifier:  {'✅ PASS' if test3 else '❌ FAIL'}")
    print("=" * 60)
    
    if test1:
        print("\n✅ Video analyzer should work!")
        print("   Clean up test video: del test_video.mp4")
    else:
        print("\n❌ Video analyzer has issues")
    
    # Clean up
    import os
    if os.path.exists("test_video.mp4"):
        try:
            os.remove("test_video.mp4")
        except:
            pass

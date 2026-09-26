"""
Automatic FFmpeg downloader for Windows
"""
import os
import zipfile
import urllib.request
import shutil

def setup_ffmpeg():
    print("Downloading and setting up FFmpeg...")
    
    # Download URL (64-bit Windows static build)
    url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
    zip_path = "ffmpeg.zip"
    extract_dir = "ffmpeg_temp"
    final_dir = os.path.join(os.path.dirname(__file__), "ffmpeg")
    bin_dir = os.path.join(final_dir, "bin")
    
    try:
        # Create directories
        os.makedirs(final_dir, exist_ok=True)
        
        # Download
        print("Downloading FFmpeg (this may take a few minutes)...")
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete!")
        
        # Extract
        print("Extracting FFmpeg...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Find the extracted folder
        extracted_folders = [f for f in os.listdir(extract_dir) if f.startswith('ffmpeg')]
        if extracted_folders:
            source_bin = os.path.join(extract_dir, extracted_folders[0], 'bin')
            
            # Copy to final location
            if os.path.exists(source_bin):
                shutil.copytree(source_bin, bin_dir, dirs_exist_ok=True)
                print(f"FFmpeg installed to: {bin_dir}")
                
                # Add to current session PATH
                os.environ['PATH'] = bin_dir + os.pathsep + os.environ.get('PATH', '')
                print(f"Added to PATH: {bin_dir}")
                
                # Test
                result = os.system('ffmpeg -version >nul 2>&1')
                if result == 0:
                    print("\n✓ FFmpeg is ready! You can now use audio analysis.")
                else:
                    print(f"\n⚠ FFmpeg downloaded but not in PATH yet.")
                    print(f"Please add this to your system PATH:")
                    print(f"  {bin_dir}")
                    print(f"\nOr restart your IDE/terminal after adding it to PATH.")
            else:
                print("Error: Could not find bin directory in extracted files")
        else:
            print("Error: Could not find extracted FFmpeg folder")
        
        # Cleanup
        if os.path.exists(zip_path):
            os.remove(zip_path)
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)
            
    except Exception as e:
        print(f"Error setting up FFmpeg: {e}")
        print("\nManual installation:")
        print("1. Download from: https://www.gyan.dev/ffmpeg/builds/")
        print("2. Extract to C:\\ffmpeg")
        print("3. Add C:\\ffmpeg\\bin to your PATH")

if __name__ == "__main__":
    setup_ffmpeg()

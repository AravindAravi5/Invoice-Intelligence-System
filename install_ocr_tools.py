"""
Invoice Intelligence System - OCR Tools Auto-Installer
Downloads and installs Tesseract and Poppler on Windows
"""

import os
import sys
import urllib.request
import zipfile
import subprocess
from pathlib import Path

def download_file(url, destination):
    """Download a file from URL to destination"""
    try:
        print(f"Downloading: {url}")
        urllib.request.urlretrieve(url, destination)
        print(f"[OK] Downloaded to: {destination}")
        return True
    except Exception as e:
        print(f"[X] Error downloading {url}: {e}")
        return False

def extract_zip(zip_path, extract_to):
    """Extract ZIP file"""
    try:
        print(f"Extracting: {zip_path}")
        os.makedirs(extract_to, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"[OK] Extracted to: {extract_to}")
        return True
    except Exception as e:
        print(f"[X] Error extracting {zip_path}: {e}")
        return False

def add_to_path(path_str):
    """Add directory to Windows PATH"""
    try:
        current_path = os.environ.get('PATH', '')
        if path_str not in current_path:
            new_path = f"{path_str};{current_path}"
            os.environ['PATH'] = new_path
            # Also try to set it permanently
            try:
                subprocess.run([
                    'reg', 'add',
                    'HKEY_CURRENT_USER\\Environment',
                    '/v', 'PATH',
                    '/d', new_path,
                    '/f'
                ], check=True, capture_output=True)
                print(f"[OK] Added to PATH: {path_str}")
            except:
                print(f"[!] Could not add to permanent PATH (run as admin for this)")
            return True
    except Exception as e:
        print(f"[X] Error adding to PATH: {e}")
        return False

def main():
    print("=" * 60)
    print("Invoice Intelligence System - OCR Tools Installer")
    print("=" * 60)
    print()
    
    # Define URLs and paths
    tesseract_url = "https://github.com/tesseract-ocr/tesseract/releases/download/5.5.0/tesseract-ocr-w64-setup-5.5.0.20241111.exe"
    poppler_url = "https://github.com/oschwartz10612/poppler-windows/releases/download/v24.08.0-0/Release-24.08.0-0.zip"
    
    base_dir = Path(__file__).parent.absolute()
    tesseract_path = base_dir / "bin" / "Tesseract-OCR"
    poppler_path = base_dir / "bin" / "poppler"
    temp_dir = base_dir / "temp_setup"
    
    # Create temp directory
    temp_dir.mkdir(exist_ok=True)
    
    # Download files
    print("Step 1: Downloading Tesseract-OCR...")
    tesseract_exe = temp_dir / "tesseract.exe"
    if not download_file(tesseract_url, str(tesseract_exe)):
        print("Failed to download Tesseract. Please download manually from:")
        print("https://github.com/UB-Mannheim/tesseract/wiki")
        return False
    
    print()
    print("Step 2: Downloading Poppler...")
    poppler_zip = temp_dir / "poppler.zip"
    if not download_file(poppler_url, str(poppler_zip)):
        print("Failed to download Poppler. Please download manually from:")
        print("https://github.com/oschwartz10612/poppler-windows/releases")
        return False
    
    print()
    print("Step 3: Installing Tesseract...")
    try:
        print(f"Running: {tesseract_exe}")
        subprocess.run([str(tesseract_exe), '/S', f'/D={tesseract_path}'], check=False)
        if tesseract_path.exists():
            print(f"[OK] Tesseract installed to: {tesseract_path}")
        else:
            print(f"[!] Tesseract installer ran but directory not found")
    except Exception as e:
        print(f"[!] Error installing Tesseract: {e}")
    
    print()
    print("Step 4: Extracting Poppler...")
    poppler_extract = temp_dir / "poppler_extract"
    if extract_zip(str(poppler_zip), str(poppler_extract)):
        # Move extracted files to Program Files
        try:
            release_dir = poppler_extract / "poppler-24.08.0"
            if not release_dir.exists():
                release_dir = poppler_extract / "Release-24.08.0-0"
            if not release_dir.exists():
                # Just find the first directory inside the extracted zip
                dirs = [d for d in poppler_extract.iterdir() if d.is_dir()]
                if dirs:
                    release_dir = dirs[0]
            if release_dir.exists():
                poppler_path.mkdir(exist_ok=True, parents=True)
                # Copy all contents
                for item in release_dir.iterdir():
                    import shutil
                    dest = poppler_path / item.name
                    if item.is_dir():
                        if dest.exists():
                            shutil.rmtree(dest)
                        shutil.copytree(item, dest)
                    else:
                        shutil.copy2(item, dest)
                print(f"[OK] Poppler extracted to: {poppler_path}")
        except Exception as e:
            print(f"[!] Error moving Poppler files: {e}")
    
    print()
    print("Step 5: Configuring PATH...")
    
    # Add to PATH
    tesseract_bin = tesseract_path
    poppler_bin = poppler_path / "Library" / "bin"
    
    add_to_path(str(tesseract_bin))
    add_to_path(str(poppler_bin))
    
    print()
    print("=" * 60)
    print("Installation Complete!")
    print("=" * 60)
    print()
    print("Installed locations:")
    print(f"  Tesseract: {tesseract_path}")
    print(f"  Poppler:   {poppler_path}")
    print()
    print("Next steps:")
    print("1. Close all terminals and VS Code")
    print("2. Reopen VS Code")
    print("3. Restart the services:")
    print("   - python app.py")
    print("   - python -m streamlit run ui.py --server.port 8501")
    print("4. Go to http://localhost:8501")
    print("5. Try uploading an invoice PDF")
    print()
    
    # Write paths to .env
    env_path = base_dir / ".env"
    with open(env_path, "w") as f:
        f.write(f"TESSERACT_PATH={tesseract_path}\\tesseract.exe\n")
        f.write(f"POPPLER_PATH={poppler_path}\\Library\\bin\n")
    print("[OK] Updated .env file with local paths")
    
    # input("Press Enter to close...")
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        # input("Press Enter to close...")
        sys.exit(1)

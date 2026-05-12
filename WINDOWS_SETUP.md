# Windows Setup Guide - Invoice Intelligence System

## ⚠️ Fix OCR Error: "tesseract is not installed or it's not in your PATH"

### Step 1: Install Tesseract-OCR

1. **Download Tesseract installer**
   - Go to: https://github.com/UB-Mannheim/tesseract/wiki
   - Download: `tesseract-ocr-w64-setup-v5.x.x.exe` (latest version)

2. **Run the installer**
   - Accept default installation path: `C:\Program Files\Tesseract-OCR`
   - Complete the installation

3. **Verify installation**
   - Open PowerShell and run:
     ```powershell
     & "C:\Program Files\Tesseract-OCR\tesseract.exe" --version
     ```
   - Should show version information

### Step 2: Install Poppler

#### Option A: Manual Download (Easiest)

1. **Download Poppler binaries**
   - Go to: https://github.com/oschwartz10612/poppler-windows/releases
   - Download: `Release-xx.xx.x.zip` (latest version)

2. **Extract to Program Files**
   - Extract ZIP to: `C:\Program Files\poppler`
   - Verify: `C:\Program Files\poppler\Library\bin\pdftoppm.exe` exists

3. **Verify installation**
   - Open PowerShell and run:
     ```powershell
     & "C:\Program Files\poppler\Library\bin\pdftoppm.exe" -v
     ```
   - Should show version information

#### Option B: Using Chocolatey

```powershell
# Install Chocolatey first if needed (run as Administrator)
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Then install packages
choco install tesseract poppler -y
```

### Step 3: Update Configuration

1. **Copy `.env.example` to `.env`** (if not already done)
   ```powershell
   cp .\.env.example .\.env
   ```

2. **Edit `.env` file** with paths:
   ```env
   TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
   POPPLER_PATH=C:\Program Files\poppler\Library\bin
   ```

   > **Note:** If you installed in different locations, update the paths accordingly

### Step 4: Restart the Application

1. **Stop the running services**
   - Press `Ctrl+C` in both terminal windows (FastAPI and Streamlit)

2. **Restart FastAPI**
   ```powershell
   python app.py
   ```

3. **In another terminal, restart Streamlit**
   ```powershell
   python -m streamlit run ui.py --server.port 8501
   ```

4. **Test the upload**
   - Go to: http://localhost:8501
   - Upload the PDF file again
   - It should now work!

---

## Troubleshooting

### "tesseract is not installed or it's not in your PATH"

**Solution 1:** Update TESSERACT_PATH in `.env`
```env
TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
```

**Solution 2:** Add to Windows PATH (Manual)
1. Search for "Environment Variables" in Windows
2. Click "Edit the system environment variables"
3. Click "Environment Variables..."
4. Under "User variables" or "System variables", click "New"
5. Variable name: `TESSERACT_PATH`
6. Variable value: `C:\Program Files\Tesseract-OCR\tesseract.exe`
7. Click OK and restart your terminal

### "poppler is not installed"

**Solution:** Update POPPLER_PATH in `.env`
```env
POPPLER_PATH=C:\Program Files\poppler\Library\bin
```

### "pdftoppm not found" or "pdfimages not found"

- Verify Poppler is extracted correctly to `C:\Program Files\poppler\Library\bin`
- Check for these files:
  - `C:\Program Files\poppler\Library\bin\pdftoppm.exe`
  - `C:\Program Files\poppler\Library\bin\pdfimages.exe`

### Verify Both Are Working

Run this PowerShell script to verify:

```powershell
Write-Host "Checking Tesseract..." -ForegroundColor Green
& "C:\Program Files\Tesseract-OCR\tesseract.exe" --version

Write-Host "`nChecking Poppler..." -ForegroundColor Green
& "C:\Program Files\poppler\Library\bin\pdftoppm.exe" -v

Write-Host "`nChecking Python modules..." -ForegroundColor Green
python -c "import pytesseract; import pdf2image; print('Python modules: OK')"
```

---

## Quick Installation Script (Automated)

Run this PowerShell script as Administrator to automate most of the process:

```powershell
# Define paths
$tesseractUrl = "https://github.com/UB-Mannheim/tesseract/releases/download/v5.3.0/tesseract-ocr-w64-setup-v5.3.0.exe"
$popplerUrl = "https://github.com/oschwartz10612/poppler-windows/releases/download/v23.08.0/Release-23.08.0.zip"

$tempDir = "$env:TEMP\invoice_setup"
$tesseractExe = "$tempDir\tesseract-setup.exe"
$popplerZip = "$tempDir\poppler.zip"

# Create temp directory
New-Item -ItemType Directory -Path $tempDir -Force | Out-Null

# Download Tesseract
Write-Host "Downloading Tesseract..." -ForegroundColor Green
(New-Object Net.WebClient).DownloadFile($tesseractUrl, $tesseractExe)

# Download Poppler
Write-Host "Downloading Poppler..." -ForegroundColor Green
(New-Object Net.WebClient).DownloadFile($popplerUrl, $popplerZip)

# Extract Poppler
Write-Host "Extracting Poppler..." -ForegroundColor Green
Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::ExtractToDirectory($popplerZip, "C:\Program Files\")
Rename-Item "C:\Program Files\Release-23.08.0" "C:\Program Files\poppler" -Force

# Run Tesseract installer
Write-Host "Running Tesseract installer..." -ForegroundColor Green
& $tesseractExe /S

Write-Host "Setup complete! Update .env with the paths above." -ForegroundColor Green
```

---

## Next Steps

After completing the setup:

1. ✅ Tesseract is installed and configured
2. ✅ Poppler is installed and configured
3. ✅ `.env` file has the correct paths
4. ✅ Restart the application
5. ✅ Try uploading an invoice again

For more help, see the main [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)

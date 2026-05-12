@echo off
REM Quick Setup Script for Tesseract and Poppler
REM Run this as Administrator

echo.
echo ====== Invoice Intelligence System - Quick Setup ======
echo.
echo This script will download and install Tesseract-OCR and Poppler
echo Required for PDF/Image processing
echo.
echo This may take 5-10 minutes depending on your internet speed.
echo.

REM Check if running as administrator
>nul 2>&1 "%SYSTEMROOT%\system32\cacls.exe" "%SYSTEMROOT%\system32\config\system"
if '%errorlevel%' NEQ '0' (
    echo ERROR: This script must be run as Administrator!
    echo Please right-click this file and select "Run as Administrator"
    pause
    exit /b 1
)

REM Create directories
echo Creating directories...
if not exist "%ProgramFiles%\Tesseract-OCR" (
    mkdir "%ProgramFiles%\Tesseract-OCR" >nul 2>&1
)
if not exist "%ProgramFiles%\poppler" (
    mkdir "%ProgramFiles%\poppler" >nul
)

REM Download Tesseract
echo.
echo Step 1: Downloading Tesseract-OCR...
if not exist "%TEMP%\tesseract-setup.exe" (
    powershell -Command "$ProgressPreference = 'SilentlyContinue'; (New-Object Net.WebClient).DownloadFile('https://github.com/UB-Mannheim/tesseract/releases/download/v5.3.0/tesseract-ocr-w64-setup-v5.3.0.exe', '%TEMP%\tesseract-setup.exe')"
    if %errorlevel% equ 0 (
        echo [OK] Tesseract downloaded
    ) else (
        echo [ERROR] Failed to download Tesseract
        pause
        exit /b 1
    )
) else (
    echo [OK] Tesseract already downloaded
)

REM Install Tesseract
echo.
echo Step 2: Installing Tesseract-OCR...
"%TEMP%\tesseract-setup.exe" /S /D="%ProgramFiles%\Tesseract-OCR"
if %errorlevel% equ 0 (
    echo [OK] Tesseract installed successfully
) else (
    echo [WARNING] Tesseract installation had issues, continuing...
)

REM Wait for installer to finish
timeout /t 3 /nobreak >nul

REM Download Poppler
echo.
echo Step 3: Downloading Poppler...
if not exist "%TEMP%\poppler.zip" (
    powershell -Command "$ProgressPreference = 'SilentlyContinue'; (New-Object Net.WebClient).DownloadFile('https://github.com/oschwartz10612/poppler-windows/releases/download/v23.08.0/Release-23.08.0.zip', '%TEMP%\poppler.zip')"
    if %errorlevel% equ 0 (
        echo [OK] Poppler downloaded
    ) else (
        echo [ERROR] Failed to download Poppler
        pause
        exit /b 1
    )
) else (
    echo [OK] Poppler already downloaded
)

REM Extract Poppler
echo.
echo Step 4: Extracting Poppler...
powershell -Command "Add-Type -AssemblyName System.IO.Compression.FileSystem; [System.IO.Compression.ZipFile]::ExtractToDirectory('%TEMP%\poppler.zip', '%ProgramFiles%\poppler_temp', $true)"
if %errorlevel% equ 0 (
    echo [OK] Poppler extracted
) else (
    echo [ERROR] Failed to extract Poppler
    pause
    exit /b 1
)

REM Move poppler files to correct location
if exist "%ProgramFiles%\poppler_temp\Release-23.08.0" (
    move "%ProgramFiles%\poppler_temp\Release-23.08.0\*" "%ProgramFiles%\poppler\" >nul 2>&1
    rmdir /s /q "%ProgramFiles%\poppler_temp\" >nul 2>&1
)

REM Verify installations
echo.
echo Step 5: Verifying installations...
if exist "%ProgramFiles%\Tesseract-OCR\tesseract.exe" (
    echo [OK] Tesseract found at: %ProgramFiles%\Tesseract-OCR\tesseract.exe
) else (
    echo [WARNING] Tesseract not found at expected location
)

if exist "%ProgramFiles%\poppler\Library\bin\pdftoppm.exe" (
    echo [OK] Poppler found at: %ProgramFiles%\poppler\Library\bin\pdftoppm.exe
) else (
    echo [WARNING] Poppler not found at expected location
)

REM Add to PATH
echo.
echo Step 6: Adding to Windows PATH...
setx PATH "%ProgramFiles%\Tesseract-OCR;%ProgramFiles%\poppler\Library\bin;%PATH%"
if %errorlevel% equ 0 (
    echo [OK] PATH updated
) else (
    echo [WARNING] Failed to update PATH
)

echo.
echo ====== Setup Complete ======
echo.
echo Next steps:
echo 1. Close all terminals and PowerShell windows
echo 2. Reopen a new PowerShell terminal
echo 3. Navigate to your project directory
echo 4. Run: python app.py
echo 5. In another terminal, run: python -m streamlit run ui.py --server.port 8501
echo 6. Go to http://localhost:8501 and try uploading an invoice
echo.
echo If you still get errors:
echo - The .env file in your project has the paths configured
echo - Check WINDOWS_SETUP.md for manual configuration steps
echo.
pause

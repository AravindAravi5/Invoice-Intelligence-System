@echo off
REM Install System Dependencies for Invoice Intelligence System
REM Requires: Windows 10+ with PowerShell

echo ====== Installing System Dependencies ======
echo.
echo This script will download and install Tesseract and Poppler
echo Required for PDF/Image processing
echo.

REM Create temp directory
if not exist "%TEMP%\invoice_deps" mkdir "%TEMP%\invoice_deps"
cd /d "%TEMP%\invoice_deps"

echo.
echo [1/3] Downloading Tesseract-OCR...
powershell -Command "(New-Object Net.WebClient).DownloadFile('https://github.com/UB-Mannheim/tesseract/wiki/Downloads', '%TEMP%\invoice_deps\tesseract_download.html')"
echo Please download Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
echo Recommended: tesseract-ocr-w64-setup-v5.x.x.exe
echo.
pause

echo.
echo [2/3] Downloading Poppler...
echo Extracting prebuilt poppler binaries...
powershell -Command ^
  "$url = 'https://github.com/oschwartz10612/poppler-windows/releases/download/v23.08.0/Release-23.08.0.zip'; " ^
  "(New-Object Net.WebClient).DownloadFile($url, '%TEMP%\invoice_deps\poppler.zip'); " ^
  "Add-Type -AssemblyName System.IO.Compression.FileSystem; " ^
  "[System.IO.Compression.ZipFile]::ExtractToDirectory('%TEMP%\invoice_deps\poppler.zip', '%TEMP%\invoice_deps\')"

echo.
if exist "%TEMP%\invoice_deps\Release-23.08.0\Library\bin" (
    echo Poppler extracted successfully
    echo Location: %TEMP%\invoice_deps\Release-23.08.0\Library\bin
) else (
    echo Poppler extraction may have failed, please verify installation
)

echo.
echo [3/3] Configuration...
echo.
echo After Tesseract installation, note the installation path (default: C:\Program Files\Tesseract-OCR)
echo.
echo Add the following to your .env file or set as environment variables:
echo TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
echo POPPLER_PATH=%TEMP%\invoice_deps\Release-23.08.0\Library\bin
echo.
echo ====== Setup Complete ======
echo Please update .env file with the paths above and restart the application
echo.
pause

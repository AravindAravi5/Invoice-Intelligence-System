@echo off
REM Invoice Intelligence System - Quick Deployment Script (Windows)

setlocal enabledelayedexpansion

echo ==== Invoice Intelligence System - Quick Deployment ====
echo.

REM Check if Docker is installed
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Docker is not installed
    echo Please install Docker Desktop from https://docs.docker.com/desktop/install/windows-install/
    pause
    exit /b 1
)

echo [OK] Docker is installed
echo.

REM Build Docker image
echo Building Docker image...
docker build -t invoice-intelligence:latest .
if %errorlevel% neq 0 (
    echo Failed to build Docker image
    pause
    exit /b 1
)

echo [OK] Docker image built successfully
echo.

REM Check if container is already running and stop it
docker ps --format "table {{.Names}}" | findstr /R "^invoice-api$" >nul
if %errorlevel% equ 0 (
    echo Stopping existing invoice-api container...
    docker stop invoice-api >nul
    docker rm invoice-api >nul
)

REM Run Docker container
echo Starting invoice-api container...
docker run -d ^
  -p 8000:8000 ^
  -v "%cd%\temp_uploads:/app/temp_uploads" ^
  --name invoice-api ^
  --restart unless-stopped ^
  invoice-intelligence:latest

if %errorlevel% neq 0 (
    echo Failed to start container
    pause
    exit /b 1
)

echo [OK] Container started successfully
echo.

REM Wait for container to be healthy
echo Waiting for application to be ready...
for /l %%i in (1,1,60) do (
    powershell -Command "try { $response = Invoke-WebRequest -Uri http://localhost:8000/docs -UseBasicParsing; exit 0 } catch { exit 1 }"
    if !errorlevel! equ 0 (
        echo [OK] Application is ready!
        goto :ready
    )
    echo.
    timeout /t 1 /nobreak >nul
)

echo Application failed to start within timeout
docker logs invoice-api
pause
exit /b 1

:ready
echo.
echo ==== Deployment Complete ====
echo.
echo API is now running at:
echo   http://localhost:8000
echo.
echo API Documentation:
echo   http://localhost:8000/docs
echo.
echo Useful commands:
echo   View logs:      docker logs -f invoice-api
echo   Stop container: docker stop invoice-api
echo   Remove image:   docker rmi invoice-intelligence:latest
echo.
pause

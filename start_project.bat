@echo off
title CardioVision AI - Project Launcher
color 0A

echo ============================================
echo   CardioVision AI - Starting All Services
echo ============================================
echo.

:: Step 1: Start Redis Server
echo [1/3] Starting Redis Server...
start "Redis Server" cmd /k "title Redis Server && color 0C && redis-server"
timeout /t 3 /nobreak >nul

:: Step 2: Start Django Development Server
echo [2/3] Starting Django Server...
start "Django Server" cmd /k "title Django Server && color 0B && cd /d E:\Major Project\Automated-Heart-Disease-Detection-Using-ECG-Image-Analysis && call venv\Scripts\activate && python manage.py runserver"
timeout /t 3 /nobreak >nul

:: Step 3: Start Celery Worker
echo [3/3] Starting Celery Worker...
start "Celery Worker" cmd /k "title Celery Worker && color 0E && cd /d E:\Major Project\Automated-Heart-Disease-Detection-Using-ECG-Image-Analysis && call venv\Scripts\activate && python -m celery -A ecg_project worker -l info --pool=threads --concurrency=2"

echo.
echo ============================================
echo   All 3 services are now running!
echo   Open browser: http://127.0.0.1:8000
echo ============================================
echo.
pause

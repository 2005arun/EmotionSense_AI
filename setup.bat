@echo off
echo ========================================
echo   EmotionSense - Interactive Setup
echo ========================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

:: Check Node.js
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js is not installed or not in PATH
    echo Please install Node.js 18+ from https://nodejs.org
    pause
    exit /b 1
)

echo [OK] Python and Node.js are installed
echo.

:: Install Python dependencies
echo [1/4] Installing Python dependencies...
pip install -r requirements-minimal.txt
if errorlevel 1 (
    echo [WARNING] Some Python packages failed to install
    echo Continuing with available packages...
)
echo.

:: Install React dependencies
echo [2/4] Installing React dependencies...
cd frontend-react
call npm install
cd ..
echo.

echo ========================================
echo   Setup Complete!
echo ========================================
echo.
echo To run the application:
echo.
echo   Option 1 - Full Interactive Mode:
echo     1. Start API:    python -m backend.api
echo     2. Start React:  cd frontend-react ^&^& npm start
echo     3. Open:         http://localhost:3000
echo.
echo   Option 2 - Streamlit Mode:
echo     streamlit run frontend/app.py
echo.
echo ========================================
pause

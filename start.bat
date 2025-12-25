@echo off
echo ========================================
echo   EmotionSense - Starting Application
echo ========================================
echo.
echo Starting Flask API server...
echo.

:: Start API in background
start "EmotionSense API" cmd /c "python -m backend.api"

:: Wait for API to start
timeout /t 3 /nobreak >nul

echo Starting React frontend...
echo.

:: Start React
cd frontend-react
start "EmotionSense Frontend" cmd /c "npm start"

echo.
echo ========================================
echo   Application is starting...
echo ========================================
echo.
echo   API:      http://localhost:5000
echo   Frontend: http://localhost:3000
echo.
echo   Close this window and the opened
echo   terminals to stop the application.
echo ========================================
echo.
pause

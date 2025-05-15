@echo off
echo Starting TopicMind for Windows
echo ============================

REM Check for Python installation
python --version > nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: Python not found! Please install Python 3.9+
    exit /b 1
)

REM Create necessary directories
mkdir logs\gpt 2> nul
mkdir logs\semantic 2> nul
mkdir logs\summaries 2> nul
mkdir logs\eval 2> nul
mkdir models\cache 2> nul

REM Check for .env file
if not exist .env (
    echo Warning: .env file not found.
    echo Creating .env.example file. Please edit with your OpenAI API key.
    copy .env.example .env 2> nul
    if not exist .env.example (
        echo @echo off > .env
        echo OPENAI_API_KEY=your_key_here >> .env
        echo MODEL_DEVICE=cpu >> .env
        echo API_TIMEOUT=180 >> .env
        echo USE_SMALL_MODELS=true >> .env
    )
)

REM Install requirements if needed
pip install -r requirements.txt

REM Start Flask backend
start /B cmd /c "python app.py > logs\backend.log 2>&1"
echo Starting backend server on port 5001...

REM Wait for backend to start
timeout /t 5 /nobreak > nul
echo Waiting for API to become healthy...

REM Check API health (retry several times)
set MAX_RETRIES=30
set RETRY_COUNT=0

:CHECK_API
curl -s http://localhost:5001/health > nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo API is healthy! Starting frontend...
    goto START_FRONTEND
) else (
    set /a RETRY_COUNT+=1
    if %RETRY_COUNT% lss %MAX_RETRIES% (
        echo API not healthy yet. Retrying in 5 seconds... (%RETRY_COUNT%/%MAX_RETRIES%)
        timeout /t 5 /nobreak > nul
        goto CHECK_API
    ) else (
        echo Failed to start API after %MAX_RETRIES% retries.
        echo Check logs\backend.log for errors.
        exit /b 1
    )
)

:START_FRONTEND
REM Start Streamlit frontend
start /B cmd /c "streamlit run frontend/streamlit_app.py --server.port=8501 > logs\frontend.log 2>&1"
echo Starting Streamlit frontend on port 8501...
timeout /t 3 /nobreak > nul

echo ============================
echo TopicMind is running!
echo - API: http://localhost:5001
echo - UI: http://localhost:8501
echo ============================
echo Press any key to stop all services...
pause > nul

REM Clean up
taskkill /F /IM python.exe > nul 2>&1
echo TopicMind stopped.
exit /b 0 
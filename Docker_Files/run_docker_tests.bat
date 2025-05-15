@echo off
setlocal EnableDelayedExpansion

echo ==========================================
echo     TopicMind Docker Compatibility Tests
echo ==========================================

REM Create logs directory if it doesn't exist
mkdir logs\tests\individual_tests 2>nul

REM Container status check
echo.
echo Checking Docker container status...
docker-compose ps | findstr "Up" >nul
if %ERRORLEVEL% equ 0 (
    echo ✅ Docker container is running.
) else (
    echo ❌ Docker container is not running.
    echo Starting Docker container...
    docker-compose up -d
    timeout /t 10 /nobreak >nul
    docker-compose ps | findstr "Up" >nul
    if %ERRORLEVEL% equ 0 (
        echo ✅ Docker container is now running.
    ) else (
        echo ❌ Failed to start Docker container. Please check docker-compose logs.
        exit /b 1
    )
)

REM Check container logs for errors
echo.
echo Checking container logs for errors...
docker-compose logs | findstr /i "error" > logs\tests\individual_tests\docker_errors.log
for /f %%i in ('type logs\tests\individual_tests\docker_errors.log ^| find /c /v ""') do set ERROR_COUNT=%%i
if !ERROR_COUNT! gtr 0 (
    echo ⚠️ Found !ERROR_COUNT! error messages in logs.
    echo See logs\tests\individual_tests\docker_errors.log for details.
) else (
    echo ✅ No errors found in container logs.
)

REM Port availability check (Windows-specific approach)
echo.
echo Checking port availability...
netstat -an | findstr "5001" | findstr "LISTENING" >nul
if %ERRORLEVEL% equ 0 (
    echo ✅ API server (port 5001) is running.
) else (
    echo ❌ API server (port 5001) is not running.
)

netstat -an | findstr "8501" | findstr "LISTENING" >nul
if %ERRORLEVEL% equ 0 (
    echo ✅ Streamlit UI (port 8501) is running.
) else (
    echo ❌ Streamlit UI (port 8501) is not running.
)

REM API health check
echo.
echo Running API tests...
set TIMEOUT_SECONDS=30
curl.exe --max-time %TIMEOUT_SECONDS% -s http://localhost:5001/health >nul
if %ERRORLEVEL% equ 0 (
    echo ✅ API health endpoint is responding.
    echo Running full API tests...
    python tests/api_test.py > logs\tests\individual_tests\api_test_results.log 2>&1
    type logs\tests\individual_tests\api_test_results.log
) else (
    echo ❌ API health endpoint is not responding (timeout after %TIMEOUT_SECONDS%s).
    echo This could be due to model loading delays.
)

REM Basic UI check
echo.
echo Running basic UI test...
curl.exe --max-time %TIMEOUT_SECONDS% -s http://localhost:8501 >nul
if %ERRORLEVEL% equ 0 (
    echo ✅ Streamlit UI is accessible.
    echo For detailed UI testing, run: python tests/ui_test.py
) else (
    echo ❌ Streamlit UI is not accessible.
)

REM Environment variable handling check
echo.
echo Checking environment variable handling...
docker-compose logs | findstr "OPENAI_API_KEY environment variable not found" >nul
if %ERRORLEVEL% equ 0 (
    echo ✅ Application correctly detects missing API key.
) else (
    echo ⚠️ No message found about missing API key.
)

REM Summary
echo.
echo ===========================================
echo           Test Summary
echo ===========================================
echo Docker container status: Running
docker-compose ps | findstr "Up" >nul
if %ERRORLEVEL% equ 0 (
    echo API server (port 5001): Running
) else (
    echo API server (port 5001): Not running
)

netstat -an | findstr "8501" | findstr "LISTENING" >nul
if %ERRORLEVEL% equ 0 (
    echo Streamlit UI (port 8501): Running
) else (
    echo Streamlit UI (port 8501): Not running
)

curl.exe --max-time %TIMEOUT_SECONDS% -s http://localhost:5001/health >nul
if %ERRORLEVEL% equ 0 (
    echo API health endpoint: Responding
) else (
    echo API health endpoint: Not responding
)

curl.exe --max-time %TIMEOUT_SECONDS% -s http://localhost:8501 >nul
if %ERRORLEVEL% equ 0 (
    echo Streamlit UI accessible: Yes
) else (
    echo Streamlit UI accessible: No
)

if !ERROR_COUNT! gtr 0 (
    echo Error messages in logs: !ERROR_COUNT!
) else (
    echo Error messages in logs: None
)

echo.
echo Complete test report available at:
echo logs\tests\individual_tests\docker_test_results.md

REM Generate timestamp
echo.
echo Test completed at: %DATE% %TIME%
endlocal 
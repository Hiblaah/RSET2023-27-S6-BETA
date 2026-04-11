@echo off
echo Installing dependencies...
call npm install
if %errorlevel% neq 0 (
    echo Error: npm install failed. Please ensure Node.js is installed.
    pause
    exit /b %errorlevel%
)
echo Dependencies installed successfully!
pause

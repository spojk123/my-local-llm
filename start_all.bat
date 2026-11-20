@echo off
chcp 65001 > nul
cd /d "%~dp0"

start "1. LangServe API (8000)" cmd /k "python server.py"
timeout /t 5 /nobreak > nul
start "2. ngrok Tunnel (8000)" cmd /k "ngrok.exe http 8000"
start "3. My LLM Web App (8501)" cmd /k "python -m streamlit run app.py"
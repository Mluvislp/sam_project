@echo off
cd /d "%~dp0"
start "SAM Text Browser" http://127.0.0.1:7862
"%~dp0venv\Scripts\python.exe" "%~dp0sam_text_local_app.py"
pause


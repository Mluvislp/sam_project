@echo off
cd /d "%~dp0"
start "SAM Browser" http://127.0.0.1:7861
"%~dp0venv\Scripts\python.exe" "%~dp0sam_local_app.py"
pause


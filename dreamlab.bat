@echo off
REM Dream Lab dashboard in its own window. Open http://localhost:8803
title Dream Lab
cd /d "%~dp0"
python dreamlab.py
pause

@echo off
title FOREX QUANT PRO TERMINAL v3.5 - Live Server (Port 8050)
cd /d "%~dp0"
echo ========================================================
echo   Uruchamianie serwera Pro Live Terminal v3.5:
echo   python gui_server.py --port 8050
echo   Adres: http://localhost:8050/
echo ========================================================
python gui_server.py --port 8050
echo.
echo [SERWER ZATRZYMANY] Nacisnij dowolny klawisz, aby zamknac to okno...
pause >nul

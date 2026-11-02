@echo off
title FOREX QUANT AI LAB - Discovery Pipeline and Dashboard
cd /d "%~dp0"
echo ========================================================
echo   KROK 1: Uruchamianie potoku badawczego:
echo   python -u run_all.py --mode standard
echo ========================================================
python -u run_all.py --mode standard

echo.
echo ========================================================
echo   KROK 2: Uruchamianie dashboardu analitycznego Streamlit:
echo   python -m streamlit run dashboard/app.py
echo ========================================================
python -m streamlit run dashboard/app.py

echo.
echo [ZAKONCZONO] Nacisnij dowolny klawisz, aby zamknac to okno...
pause >nul

@echo off
title FOREX QUANT AI LAB - ULTRA HPC Full Suite
cd /d "%~dp0"

:: Konfiguracja przebiegu ULTRA (mozesz łatwo zmienic czas w godzinach)
set DURATION_HOURS=2.0
set N_NULLS=50

echo ========================================================
echo   KROK 1: Uruchamianie potoku badawczego (TRYB ULTRA):
echo   python -u run_all.py --mode ultra --duration-hours %DURATION_HOURS% --n-nulls %N_NULLS% --enable-tier3
echo ========================================================
python -u run_all.py --mode ultra --duration-hours %DURATION_HOURS% --n-nulls %N_NULLS% --enable-tier3

echo.
echo ========================================================
echo   KROK 2: Kompilacja sygnalow dla 56 rynkow i budowa GUI:
echo ========================================================
python -u compile_timeframe_payload.py
python -u build_gui.py

echo.
echo ========================================================
echo   KROK 3: Uruchamianie Pro Live Terminal (port 8050)
echo           oraz dashboardu Streamlit (port 8501):
echo ========================================================
start "TRFX Pro Terminal Server" cmd /c "python -u gui_server.py --port 8050"
python -m streamlit run dashboard/app.py

echo.
echo [ZAKONCZONO] Nacisnij dowolny klawisz, aby zamknac to okno...
pause >nul
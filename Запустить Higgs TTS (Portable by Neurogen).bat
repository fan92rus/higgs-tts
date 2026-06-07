@echo off
chcp 65001
setlocal EnableExtensions
title Higgs Audio v3 TTS - Portable by Neurogen
cd /d "%~dp0"

set "VENV=%~dp0runtime\Scripts\python.exe"
if not exist "%VENV%" goto :noenv

echo Запуск Higgs Audio v3 TTS - Portable by Neurogen ...
echo (Первый запуск скачивает веса ~9.3 ГБ, это может занять время.)
echo.
"%VENV%" "%~dp0src\main.py"
if errorlevel 1 goto :err
exit /b 0

:noenv
echo [!] Окружение не найдено.
echo     Сначала запустите "Установка (первый запуск).bat".
echo.
pause
exit /b 1

:err
echo.
echo [!] Программа завершилась с ошибкой (см. сообщения выше).
echo.
pause
exit /b 1

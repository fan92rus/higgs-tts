@echo off
chcp 65001
setlocal EnableExtensions
title Higgs Audio v3 TTS - Portable by Neurogen (установка)
cd /d "%~dp0"

echo ============================================================
echo   Higgs Audio v3 TTS - Portable by Neurogen
echo   Установка (первый запуск)
echo ============================================================
echo.

REM --- Python: вложенный python\, затем 3.11/3.12/3.10. 3.13/3.14 НЕ подходят ---
set "BUNDLED=%~dp0python\python.exe"
set "PYKIND="
if exist "%BUNDLED%" set "PYKIND=bundled"
if not defined PYKIND ( py -3.11 --version && set "PYKIND=py311" )
if not defined PYKIND ( py -3.12 --version && set "PYKIND=py312" )
if not defined PYKIND ( py -3.10 --version && set "PYKIND=py310" )
if defined PYKIND goto :pyok
python -c "import sys;v=sys.version_info;sys.exit(0 if v.major==3 and v.minor in (10,11,12) else 1)" && set "PYKIND=python"
:pyok
if not defined PYKIND goto :nopython
echo Использую Python: %PYKIND%
echo.

echo Создаю изолированное окружение (папка runtime)...
if exist "%~dp0runtime\Scripts\python.exe" goto :haveenv
if "%PYKIND%"=="bundled" "%BUNDLED%" -m venv "%~dp0runtime"
if "%PYKIND%"=="py311"   py -3.11 -m venv "%~dp0runtime"
if "%PYKIND%"=="py312"   py -3.12 -m venv "%~dp0runtime"
if "%PYKIND%"=="py310"   py -3.10 -m venv "%~dp0runtime"
if "%PYKIND%"=="python"  python -m venv "%~dp0runtime"
:haveenv
if not exist "%~dp0runtime\Scripts\python.exe" goto :venvfail
set "VENV=%~dp0runtime\Scripts\python.exe"

REM --- Доверенные хосты для pip (обход перехвата HTTPS антивирусом/прокси) ---
set "PIPHOSTS=--trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host download.pytorch.org --trusted-host download-r2.pytorch.org"

echo Обновляю pip...
"%VENV%" -m pip install %PIPHOSTS% --upgrade pip wheel setuptools

echo.
echo  Выберите вариант установки:
echo    [1] NVIDIA GPU (авто-подбор CUDA 12.1 / 12.8 под вашу карту) - рекомендуется
echo    [2] Только CPU (без видеокарты)
echo.
set "DEV=1"
set /p "DEV=Введите 1 или 2 и нажмите Enter [1]: "

if "%DEV%"=="2" goto :cpuinstall

REM --- GPU: определяем нужный индекс PyTorch по архитектуре видеокарты ---
set "TORCH_INDEX="
for /f "usebackq delims=" %%i in (`"%VENV%" "%~dp0src\detect_cuda.py"`) do set "TORCH_INDEX=%%i"
if not defined TORCH_INDEX set "TORCH_INDEX=https://download.pytorch.org/whl/cu121"
echo Индекс PyTorch для вашей видеокарты: %TORCH_INDEX%
echo Устанавливаю PyTorch (GPU)...
"%VENV%" -m pip install %PIPHOSTS% torch torchaudio --index-url %TORCH_INDEX%
goto :aftertorch

:cpuinstall
echo Устанавливаю PyTorch (CPU)...
"%VENV%" -m pip install %PIPHOSTS% torch torchaudio --index-url https://download.pytorch.org/whl/cpu

:aftertorch
if errorlevel 1 goto :piperr

echo Устанавливаю зависимости...
"%VENV%" -m pip install %PIPHOSTS% -r "%~dp0requirements.txt"
if errorlevel 1 goto :piperr

if not "%DEV%"=="2" (
  echo Устанавливаю bitsandbytes (квантизация 8/4-bit)...
  "%VENV%" -m pip install %PIPHOSTS% bitsandbytes
)

echo.
echo ============================================================
echo   Готово! Теперь запускайте программу файлом:
echo   "Запустить Higgs TTS (Portable by Neurogen).bat"
echo.
echo   Первый запуск скачает веса (~9.3 ГБ) и выполнит
echo   однократную конвертацию. Дальше запуск мгновенный.
echo ============================================================
echo.
pause
exit /b 0

:nopython
echo [!] Не найден совместимый Python (нужен 3.10 / 3.11 / 3.12, 64-bit).
echo     В этой сборке Python должен лежать в папке "python" - похоже,
echo     её удалили или скачан неполный архив.
echo     Решение: скачайте ПОЛНЫЙ архив заново (там есть папка python),
echo     либо поставьте Python 3.11 x64 с python.org (галочка Add to PATH).
echo     ВНИМАНИЕ: Python 3.13 и 3.14 пока НЕ подходят (нет колёс PyTorch).
echo.
pause
exit /b 1

:venvfail
echo [!] Не удалось создать окружение (runtime).
echo     Желательно, чтобы путь к папке не содержал кириллицы.
echo.
pause
exit /b 1

:piperr
echo.
echo [!] Ошибка при установке пакетов (см. сообщения выше).
echo     Если в ошибке упоминается SSL / certificate - это антивирус или
echo     прокси перехватывает HTTPS. Попробуйте временно отключить
echo     проверку HTTPS в антивирусе и запустить установку снова.
echo     Также проверьте интернет-соединение.
echo.
pause
exit /b 1

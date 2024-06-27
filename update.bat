@echo off

set DIR=%~dp0system

set PATH=%DIR%\git\bin;%DIR%\python;%DIR%\python\Scripts;%PATH%
set PY_LIBS=%DIR%\python\Scripts\Lib;%DIR%\python\Scripts\Lib\site-packages
set PY_PIP=%DIR%\python\Scripts
set SKIP_VENV=1
set PIP_INSTALLER_LOCATION=%DIR%\python\get-pip.py
set TRANSFORMERS_CACHE=%DIR%\transformers-cache

git pull 2>NUL
if %ERRORLEVEL% == 0 goto :done

git reset --hard
git pull

:done
pause

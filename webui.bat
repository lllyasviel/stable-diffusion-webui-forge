@echo off

REM Detect CUDA version using nvcc and set UV_TORCH_BACKEND accordingly
where nvcc >nul 2>nul
if %ERRORLEVEL%==0 (
    for /f "tokens=2 delims== " %%v in ('nvcc --version ^| findstr /R /C:"release"') do set CUDA_VER=%%v
    set CUDA_VER=%CUDA_VER:.=%
    if "%CUDA_VER%"=="118" set UV_TORCH_BACKEND=cu118
    if "%CUDA_VER%"=="121" set UV_TORCH_BACKEND=cu121
    if "%CUDA_VER%"=="124" set UV_TORCH_BACKEND=cu124
    if not defined UV_TORCH_BACKEND set UV_TORCH_BACKEND=auto
) else (
    set UV_TORCH_BACKEND=auto
)
set UV_TORCH_BACKEND=%UV_TORCH_BACKEND%

REM Detect uv and set UV_PIP and UV_VENV if available
where uv >nul 2>nul
if %ERRORLEVEL%==0 (
    set "UV_PIP=uv pip"
    set "UV_VENV=uv venv"
    set "UV_RUN=uv run"
    set "USING_UV=1"
) else (
    set "UV_PIP=python -m pip"
    set "UV_VENV="
    set "UV_RUN="
    set "USING_UV=0"
)

if exist webui.settings.bat (
    call webui.settings.bat
)

if not defined PYTHON (set PYTHON=python)
if defined GIT (set "GIT_PYTHON_GIT_EXECUTABLE=%GIT%")
if not defined VENV_DIR (set "VENV_DIR=%~dp0%venv")

set SD_WEBUI_RESTART=tmp/restart
set ERROR_REPORTING=FALSE

mkdir tmp 2>NUL

%PYTHON% -c "" >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :check_pip
echo Couldn't launch python
goto :show_stdout_stderr

:check_pip
%PYTHON% -mpip --help >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :start_venv
if "%PIP_INSTALLER_LOCATION%" == "" goto :show_stdout_stderr
%PYTHON% "%PIP_INSTALLER_LOCATION%" >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :start_venv
echo Couldn't install pip
goto :show_stdout_stderr

:start_venv
if ["%VENV_DIR%"] == ["-"] goto :skip_venv
if ["%SKIP_VENV%"] == ["1"] goto :skip_venv

dir "%VENV_DIR%\Scripts\Python.exe" >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :activate_venv

REM Use uv venv if available, otherwise fallback to python -m venv
if defined UV_VENV (
    echo Creating venv in directory %VENV_DIR% using uv
    %UV_VENV% "%VENV_DIR%" >tmp/stdout.txt 2>tmp/stderr.txt
    set "USING_UV=1"
) else (
    for /f "delims=" %%i in ('CALL %PYTHON% -c "import sys; print(sys.executable)"') do set PYTHON_FULLNAME="%%i"
    echo Creating venv in directory %VENV_DIR% using python %PYTHON_FULLNAME%
    %PYTHON_FULLNAME% -m venv "%VENV_DIR%" >tmp/stdout.txt 2>tmp/stderr.txt
    set "USING_UV=0"
)
if %ERRORLEVEL% == 0 goto :upgrade_pip
echo Unable to create venv in directory "%VENV_DIR%"
goto :show_stdout_stderr

:upgrade_pip
if "%USING_UV%"=="1" goto :skip_venv
REM Use uv pip or pip to upgrade pip
if defined UV_PIP (
    %UV_PIP% install --upgrade pip
) else (
    "%VENV_DIR%\Scripts\Python.exe" -m pip install --upgrade pip
)
if %ERRORLEVEL% == 0 goto :activate_venv
echo Warning: Failed to upgrade PIP version

:activate_venv
if "%USING_UV%"=="1" goto :skip_venv
set PYTHON="%VENV_DIR%\Scripts\Python.exe"
call "%VENV_DIR%\Scripts\activate.bat"
echo venv %PYTHON%

:skip_venv
if [%ACCELERATE%] == ["True"] goto :accelerate
if "%USING_UV%"=="1" goto :launch_uv
goto :launch

:launch_uv
REM Use uv run to launch the app if using uv venv
%UV_RUN% python launch.py %*
if EXIST tmp/restart goto :skip_venv
pause
exit /b

:launch
%PYTHON% launch.py %*
if EXIST tmp/restart goto :skip_venv
pause
exit /b

:accelerate_launch
echo Accelerating
%ACCELERATE% launch --num_cpu_threads_per_process=6 launch.py
if EXIST tmp/restart goto :skip_venv
pause
exit /b

:show_stdout_stderr

echo.
echo exit code: %errorlevel%

for /f %%i in ("tmp\stdout.txt") do set size=%%~zi
if %size% equ 0 goto :show_stderr
echo.
echo stdout:
type tmp\stdout.txt

:show_stderr
for /f %%i in ("tmp\stderr.txt") do set size=%%~zi
if %size% equ 0 goto :show_stderr
echo.
echo stderr:
type tmp\stderr.txt

:endofscript

echo.
echo Launch unsuccessful. Exiting.
pause

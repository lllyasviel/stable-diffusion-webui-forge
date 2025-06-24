@echo off

set PYTHON=
set GIT=
set VENV_DIR=-
set COMMANDLINE_ARGS=--cuda-stream --cuda-malloc

REM Detect CUDA version using nvcc and set UV_TORCH_BACKEND accordingly
where nvcc >nul 2>nul
if %ERRORLEVEL%==0 (
    for /f "tokens=2 delims== " %%v in ('nvcc --version ^| findstr /R /C:"release"') do set CUDA_VER=%%v
    set CUDA_VER=%CUDA_VER:.=%
    if "%CUDA_VER%"=="118" set UV_TORCH_BACKEND=cu118
    if "%CUDA_VER%"=="121" set UV_TORCH_BACKEND=cu121
    if "%CUDA_VER%"=="124" set UV_TORCH_BACKEND=cu124
    if "%CUDA_VER%"=="126" set UV_TORCH_BACKEND=cu126
    if not defined UV_TORCH_BACKEND set UV_TORCH_BACKEND=auto
) else (
    set UV_TORCH_BACKEND=auto
)
set UV_TORCH_BACKEND=%UV_TORCH_BACKEND%

@REM Uncomment following code to reference an existing A1111 checkout.
@REM set A1111_HOME=Your A1111 checkout dir
@REM
@REM set VENV_DIR=%A1111_HOME%/venv
@REM set COMMANDLINE_ARGS=%COMMANDLINE_ARGS% ^
@REM  --ckpt-dir %A1111_HOME%/models/Stable-diffusion ^
@REM  --hypernetwork-dir %A1111_HOME%/models/hypernetworks ^
@REM  --embeddings-dir %A1111_HOME%/embeddings ^
@REM  --lora-dir %A1111_HOME%/models/Lora

call webui.bat

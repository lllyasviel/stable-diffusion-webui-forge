@echo off

set PYTHON=
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS=--xformers --opt-sdp-attention --disable-gpu-warning --cuda-malloc --cuda-stream --pin-shared-memory --opt-channelslast --disable-nan-check --api --listen --autolaunch

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

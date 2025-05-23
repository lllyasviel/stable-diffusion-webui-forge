@echo off

set PYTHON=C:/Users/mimic/AppData/Local/Programs/Python/Python310/python.exe
set GIT=C:/Users/mimic/AppData/Local/Programs/Python/Python310/python.exe
set A1111_HOME=D:/webui_forge_cu121_torch21/webui
set VENV_DIR=%A1111_HOME%/venv
set COMMANDLINE_ARGS= --forge-ref-a1111-home A1111_HOME

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

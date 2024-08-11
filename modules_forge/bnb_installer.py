import pkg_resources

from modules.launch_utils import run_pip

target_bitsandbytes_version = '0.43.3'


def try_install_bnb():
    try:
        bitsandbytes_version = pkg_resources.get_distribution('bitsandbytes').version
    except Exception:
        bitsandbytes_version = None

    try:
        if bitsandbytes_version != target_bitsandbytes_version:
            run_pip(
                f"install -U bitsandbytes=={target_bitsandbytes_version}",
                f"bitsandbytes=={target_bitsandbytes_version}",
            )
    except Exception as e:
        print(f'Cannot install bitsandbytes. Skipped.')

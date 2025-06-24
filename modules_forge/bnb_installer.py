import pkg_resources

target_bitsandbytes_version = '0.45.3'


def try_install_bnb():
    try:
        bitsandbytes_version = pkg_resources.get_distribution('bitsandbytes').version
    except Exception:
        bitsandbytes_version = None

    # Installation is now handled elsewhere; just warn if not correct version
    if bitsandbytes_version != target_bitsandbytes_version:
        print(f"Warning: bitsandbytes version {bitsandbytes_version} found, but {target_bitsandbytes_version} is required.")
        print("Please install the correct version manually if you encounter issues.")

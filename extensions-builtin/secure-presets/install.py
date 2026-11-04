import launch

# Secure Presets needs the 'cryptography' package (AES-256-GCM + scrypt).
if not launch.is_installed("cryptography"):
    launch.run_pip("install cryptography", "cryptography (for Secure Presets)")

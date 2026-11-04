"""Encrypted storage for prompts and presets.

Everything that contains prompt text is stored only as ciphertext:
AES-256-GCM, key derived from a passphrase with scrypt. The passphrase is never
written to disk; the derived key lives in memory only while the vault is unlocked.

Files (in <data_path>/secure_presets/):
    vault.json   scrypt parameters + salt + encrypted preset library
"""
import base64
import json
import os
import secrets
import threading
import time

MIN_PASSPHRASE_LEN = 12
_SCRYPT = {"n": 2 ** 17, "r": 8, "p": 1}
_NONCE_LEN = 12

_lock = threading.RLock()
_key = None            # bytearray while unlocked
_dir_override = None


class VaultError(Exception):
    pass


class WrongPassphrase(VaultError):
    pass


def set_dir(path):
    """Override the storage directory (used by tests)."""
    global _dir_override
    _dir_override = path


def _dir():
    if _dir_override:
        return _dir_override
    from modules import paths
    return os.path.join(paths.data_path, "secure_presets")


def _vault_path():
    return os.path.join(_dir(), "vault.json")


def _crypto():
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
        from cryptography.exceptions import InvalidTag
    except ImportError as e:
        raise VaultError("The 'cryptography' package is not installed. Restart Forge so it can install it.") from e
    return AESGCM, Scrypt, InvalidTag


def crypto_available():
    try:
        _crypto()
        return True
    except VaultError:
        return False


def _b64e(b):
    return base64.b64encode(b).decode("ascii")


def _b64d(s):
    return base64.b64decode(s.encode("ascii"))


def _derive(passphrase, salt, n, r, p):
    _, Scrypt, _ = _crypto()
    return bytearray(Scrypt(salt=salt, length=32, n=n, r=r, p=p).derive(passphrase.encode("utf-8")))


def _encrypt(key, plaintext, aad):
    AESGCM, _, _ = _crypto()
    nonce = secrets.token_bytes(_NONCE_LEN)
    return nonce + AESGCM(bytes(key)).encrypt(nonce, plaintext, aad)


def _decrypt(key, blob, aad):
    AESGCM, _, InvalidTag = _crypto()
    try:
        return AESGCM(bytes(key)).decrypt(blob[:_NONCE_LEN], blob[_NONCE_LEN:], aad)
    except InvalidTag as e:
        raise WrongPassphrase("Wrong passphrase, or the file was modified.") from e


def _atomic_write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _header_bytes(header):
    return b"vault-v1|" + json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _read_vault():
    with open(_vault_path(), "r", encoding="utf-8") as f:
        doc = json.load(f)
    return doc["header"], _b64d(doc["blob"])


def _write_vault(header, key, presets):
    plaintext = json.dumps({"presets": presets}).encode("utf-8")
    blob = _encrypt(key, plaintext, _header_bytes(header))
    _atomic_write(_vault_path(), json.dumps({"header": header, "blob": _b64e(blob)}).encode("utf-8"))


def _new_header():
    return {"v": 1, "kdf": "scrypt", **_SCRYPT, "salt": _b64e(secrets.token_bytes(16))}


def _check_passphrase(passphrase):
    if len(passphrase) < MIN_PASSPHRASE_LEN:
        raise VaultError(f"Passphrase must be at least {MIN_PASSPHRASE_LEN} characters.")


def vault_exists():
    return os.path.isfile(_vault_path())


def is_unlocked():
    return _key is not None


def _require_unlocked():
    if _key is None:
        raise VaultError("Vault is locked.")


def _load_presets():
    _require_unlocked()
    header, blob = _read_vault()
    return json.loads(_decrypt(_key, blob, _header_bytes(header)).decode("utf-8"))["presets"]


def create_vault(passphrase):
    global _key
    with _lock:
        if vault_exists():
            raise VaultError("A vault already exists.")
        _check_passphrase(passphrase)
        header = _new_header()
        key = _derive(passphrase, _b64d(header["salt"]), header["n"], header["r"], header["p"])
        _write_vault(header, key, {})
        _key = key


def unlock(passphrase):
    global _key
    with _lock:
        if not vault_exists():
            raise VaultError("No vault exists yet.")
        header, blob = _read_vault()
        key = _derive(passphrase, _b64d(header["salt"]), header["n"], header["r"], header["p"])
        try:
            _decrypt(key, blob, _header_bytes(header))
        except WrongPassphrase:
            time.sleep(1.0)  # slow down guessing through the UI
            raise
        _key = key


def lock():
    global _key
    with _lock:
        if _key is not None:
            for i in range(len(_key)):
                _key[i] = 0
        _key = None


def list_names():
    with _lock:
        return sorted(_load_presets().keys(), key=str.lower)


def get_preset(name):
    with _lock:
        presets = _load_presets()
        if name not in presets:
            raise VaultError("Preset not found.")
        return presets[name]


def save_preset(name, ui, values, tab=None):
    name = (name or "").strip()
    if not name or len(name) > 80:
        raise VaultError("Preset name must be 1-80 characters.")
    with _lock:
        presets = _load_presets()
        presets[name] = {"tab": tab, "ui": ui, "values": values, "saved": time.strftime("%Y-%m-%d %H:%M:%S")}
        header, _ = _read_vault()
        _write_vault(header, _key, presets)
    return name


def delete_preset(name):
    with _lock:
        presets = _load_presets()
        if name not in presets:
            raise VaultError("Preset not found.")
        del presets[name]
        header, _ = _read_vault()
        _write_vault(header, _key, presets)


def change_passphrase(old, new):
    global _key
    with _lock:
        _require_unlocked()
        _check_passphrase(new)
        header, blob = _read_vault()
        old_key = _derive(old, _b64d(header["salt"]), header["n"], header["r"], header["p"])
        presets = json.loads(_decrypt(old_key, blob, _header_bytes(header)).decode("utf-8"))["presets"]
        new_header = _new_header()
        new_key = _derive(new, _b64d(new_header["salt"]), new_header["n"], new_header["r"], new_header["p"])
        _write_vault(new_header, new_key, presets)
        lock()
        _key = new_key


def wipe_plaintext(path):
    """Overwrite a plaintext file with zeros, then delete it. Never reads the file.
    (SSD wear-leveling means this cannot be guaranteed to destroy every physical copy.)"""
    try:
        if not os.path.isfile(path):
            return False
        size = os.path.getsize(path)
        with open(path, "r+b") as f:
            f.write(b"\0" * size)
            f.flush()
            os.fsync(f.fileno())
        os.remove(path)
        return True
    except OSError:
        return False

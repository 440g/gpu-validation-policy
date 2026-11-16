"""Download the Seren job trace (AcmeTrace, CC BY 4.0) pinned to a Hugging Face commit and verify it."""
import hashlib
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpuval import config as C  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    C.DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not C.DATA_PATH.exists():
        print(f"downloading {C.DATA_URL}")
        urllib.request.urlretrieve(C.DATA_URL, C.DATA_PATH)
    digest = sha256(C.DATA_PATH)
    if digest != C.DATA_SHA256:
        sys.exit(f"sha256 mismatch for {C.DATA_PATH}: {digest}")
    print(f"ok: {C.DATA_PATH} ({digest[:12]})")


if __name__ == "__main__":
    main()

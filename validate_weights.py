import argparse
import hashlib
import sys
import zipfile
from pathlib import Path


DEFAULT_ARCH = "timm/swinv2_large_window12to24_192to384.ms_in22k_ft_in1k"


def compute_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def precheck_file(path: Path) -> None:
    with path.open("rb") as f:
        head = f.read(200)

    if head.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise RuntimeError(
            "This .pth is a Git LFS pointer file, not the real model weights."
        )

    if head.startswith(b"PK"):
        try:
            with zipfile.ZipFile(path, "r") as zf:
                zf.namelist()
        except zipfile.BadZipFile as exc:
            raise RuntimeError(
                "This .pth file is corrupted or incomplete: ZIP central directory is missing."
            ) from exc


def unwrap_checkpoint(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Checkpoint could not be parsed into a state_dict.")
    return checkpoint


def strip_module_prefix(state_dict):
    first_key = next(iter(state_dict.keys()))
    if first_key.startswith("module."):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def load_state_dict(path: Path):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed in the current environment."
        ) from exc

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        mode = "weights_only=True"
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
        mode = "legacy torch.load"
    except Exception:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        mode = "weights_only=False fallback"

    state_dict = unwrap_checkpoint(checkpoint)
    state_dict = strip_module_prefix(state_dict)
    return state_dict, mode


def validate_model_load(state_dict, arch: str, num_classes: int):
    try:
        import timm
    except ImportError as exc:
        raise RuntimeError(
            "timm is not installed in the current environment."
        ) from exc

    model = timm.create_model(arch, pretrained=False, num_classes=num_classes)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return missing, unexpected


def main():
    parser = argparse.ArgumentParser(description="Validate local .pth model weights.")
    parser.add_argument("weights", help="Path to the .pth file")
    parser.add_argument("--arch", default=DEFAULT_ARCH, help="Model architecture for timm.create_model")
    parser.add_argument("--num-classes", type=int, default=4, help="Number of output classes")
    parser.add_argument("--skip-model", action="store_true", help="Only test file integrity and torch.load")
    args = parser.parse_args()

    path = Path(args.weights).expanduser().resolve()
    if not path.exists():
        print(f"[FAIL] File not found: {path}")
        return 1

    print(f"[INFO] File: {path}")
    print(f"[INFO] Size: {path.stat().st_size} bytes")
    print(f"[INFO] SHA-256: {compute_sha256(path)}")

    try:
        precheck_file(path)
        print("[OK] File precheck passed")
    except Exception as exc:
        print(f"[FAIL] File precheck failed: {exc}")
        return 2

    try:
        state_dict, mode = load_state_dict(path)
        print(f"[OK] torch.load passed via {mode}")
        print(f"[INFO] state_dict keys: {len(state_dict)}")
    except Exception as exc:
        print(f"[FAIL] torch.load failed: {exc}")
        return 3

    if args.skip_model:
        return 0

    try:
        missing, unexpected = validate_model_load(state_dict, args.arch, args.num_classes)
        if missing or unexpected:
            print("[WARN] Model loaded with non-strict mismatches")
            print(f"[WARN] Missing keys: {len(missing)}")
            print(f"[WARN] Unexpected keys: {len(unexpected)}")
        else:
            print("[OK] Model structure matched exactly")
        return 0
    except Exception as exc:
        print(f"[FAIL] Model instantiation/load failed: {exc}")
        return 4


if __name__ == "__main__":
    sys.exit(main())

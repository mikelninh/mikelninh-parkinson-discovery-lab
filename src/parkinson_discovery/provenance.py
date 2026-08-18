from __future__ import annotations

import hashlib
import json
import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import Iterable


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return sha256_bytes(encoded)


def environment_fingerprint() -> dict[str, object]:
    packages: dict[str, str] = {}
    for package in ["numpy", "pandas", "scikit-learn", "rdkit", "requests"]:
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = "not-installed"
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
    }


def artifact_hashes(base_dir: Path, paths: Iterable[Path | str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for item in paths:
        path = Path(item)
        if not path.is_absolute():
            path = base_dir / path
        if path.exists() and path.is_file():
            hashes[str(path.relative_to(base_dir))] = sha256_file(path)
    return dict(sorted(hashes.items()))


def verify_artifact_hashes(base_dir: Path, expected: dict[str, str]) -> dict[str, object]:
    missing: list[str] = []
    mismatched: list[dict[str, str]] = []
    verified: list[str] = []
    for relative, expected_hash in sorted(expected.items()):
        path = base_dir / relative
        if not path.exists():
            missing.append(relative)
            continue
        actual = sha256_file(path)
        if actual != expected_hash:
            mismatched.append({"path": relative, "expected": expected_hash, "actual": actual})
        else:
            verified.append(relative)
    return {
        "ok": not missing and not mismatched,
        "verified": verified,
        "missing": missing,
        "mismatched": mismatched,
    }


def verify_manifest(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    hashes = payload.get("artifact_sha256") or payload.get("artifacts_sha256") or {}
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError(f"Manifest {path} does not contain artifact hashes")
    result = verify_artifact_hashes(path.parent, hashes)

    expected_manifest_hash = payload.get("manifest_sha256")
    if expected_manifest_hash:
        unsigned = dict(payload)
        unsigned.pop("manifest_sha256", None)
        actual_manifest_hash = canonical_json_sha256(unsigned)
        manifest_hash_ok = actual_manifest_hash == expected_manifest_hash
    else:
        actual_manifest_hash = None
        manifest_hash_ok = None

    result["artifact_hashes_ok"] = result["ok"]
    result["manifest_hash_ok"] = manifest_hash_ok
    result["manifest_sha256_actual"] = actual_manifest_hash
    result["ok"] = bool(result["ok"] and manifest_hash_ok is not False)
    result["manifest"] = str(path)
    result["manifest_type"] = payload.get("manifest_type", "unknown")
    result["snapshot_id"] = payload.get("snapshot_id")
    result["run_id"] = payload.get("run_id")
    return result

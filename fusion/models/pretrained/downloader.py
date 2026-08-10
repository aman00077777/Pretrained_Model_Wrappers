"""
Task P2 — Pretrained Model Downloader

Downloads a model's full weight snapshot from Hugging Face Hub into a local
cache directory, then performs a lightweight sanity check on any LFS-tracked
files detected via ``.gitattributes``.

Consumers::

    from fusion.models.pretrained import download_pretrained

    local_path = download_pretrained("bert-base-uncased", cache_dir="/tmp/hf")
"""
from __future__ import annotations

import os
import pathlib
from typing import List, Optional

from fusion.exceptions import FusionError
from fusion.utils.logging import get_logger

logger = get_logger(__name__)

# Minimum expected file size (bytes) for a legitimate model weight file.
# LFS pointer stubs are ~130 bytes; anything below this threshold on a file
# that *should* be a weight binary is a sign of an incomplete download.
_LFS_POINTER_MAX_BYTES: int = 512


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class DownloadError(FusionError):
    """Raised when a model snapshot download fails or is incomplete.

    Args:
        message (str): Human-readable description of the failure.
        details (Optional[dict]): Structured diagnostic payload
            (e.g. ``{"repo_id": "...", "cache_dir": "..."}``).
    """

    def __init__(self, message: str, details: Optional[dict] = None) -> None:
        super().__init__(message, details)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    """Create *path* (and any parents) if it does not already exist."""
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    logger.debug("event=cache_dir_ensured path=%s", path)


def _read_lfs_patterns(snapshot_dir: str) -> List[str]:
    """Parse ``.gitattributes`` in *snapshot_dir* and return LFS glob patterns.

    Returns an empty list when the file is absent or contains no LFS entries.
    """
    gitattributes = os.path.join(snapshot_dir, ".gitattributes")
    if not os.path.isfile(gitattributes):
        return []

    patterns: List[str] = []
    try:
        with open(gitattributes, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                # LFS lines look like: "*.bin filter=lfs diff=lfs merge=lfs -text"
                if "filter=lfs" in line:
                    pattern = line.split()[0]
                    patterns.append(pattern)
    except OSError as exc:
        logger.warning(
            "event=gitattributes_read_failed path=%s error_type=%s",
            gitattributes, type(exc).__name__,
        )
    return patterns


def _is_lfs_pointer(filepath: str) -> bool:
    """Return True if *filepath* looks like an un-fetched LFS pointer stub."""
    try:
        size = os.path.getsize(filepath)
        if size > _LFS_POINTER_MAX_BYTES:
            return False
        with open(filepath, "rb") as fh:
            header = fh.read(48)
        return header.startswith(b"version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False


def _verify_lfs_files(snapshot_dir: str, lfs_patterns: List[str]) -> None:
    """Walk *snapshot_dir* and warn about any apparent LFS pointer stubs.

    Does **not** raise — a warning is sufficient so partial caches don't
    block the caller entirely.
    """
    import fnmatch

    suspicious: List[str] = []
    for root, _dirs, files in os.walk(snapshot_dir):
        for fname in files:
            rel = os.path.relpath(os.path.join(root, fname), snapshot_dir)
            matched = any(fnmatch.fnmatch(fname, pat) for pat in lfs_patterns)
            if matched:
                full = os.path.join(root, fname)
                if _is_lfs_pointer(full):
                    suspicious.append(rel)

    if suspicious:
        logger.warning(
            "event=checksum_verification_failed "
            "snapshot_dir=%s suspicious_files=%s "
            "hint=files_may_be_lfs_pointer_stubs",
            snapshot_dir, suspicious,
        )
    else:
        logger.info(
            "event=checksum_verified snapshot_dir=%s lfs_file_count=%d",
            snapshot_dir, len(lfs_patterns),
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_pretrained(repo_id: str, cache_dir: str) -> str:
    """Download a full model snapshot from Hugging Face Hub.

    Uses :func:`huggingface_hub.snapshot_download` to fetch all files for
    *repo_id* into *cache_dir*.  After downloading, the function checks for a
    ``.gitattributes`` file to identify LFS-tracked assets and performs a
    lightweight sanity check (size + magic-byte probe) on each of them.  A
    warning is logged — not an exception — when a file looks like an
    un-fetched LFS pointer stub, so callers are notified without hard-failing.

    Args:
        repo_id: Hugging Face repository ID, e.g. ``"bert-base-uncased"`` or
            ``"openai/clip-vit-base-patch32"``.
        cache_dir: Local directory where the snapshot will be stored.
            Created automatically if it does not exist.

    Returns:
        Absolute path to the downloaded snapshot directory (the value returned
        by :func:`huggingface_hub.snapshot_download`).

    Raises:
        DownloadError: On network failures, repo-not-found (404), or any other
            error raised by the Hub client.

    Example::

        path = download_pretrained(
            "bert-base-uncased", cache_dir="/tmp/hf_cache"
        )
    """
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import RepositoryNotFoundError, EntryNotFoundError
    except ImportError as exc:
        raise DownloadError(
            "huggingface_hub is required for download_pretrained. "
            "Install it with: pip install huggingface_hub",
            details={"error_type": "ImportError"},
        ) from exc

    _ensure_dir(cache_dir)

    logger.info(
        "event=download_started repo_id=%s cache_dir=%s",
        repo_id, cache_dir,
    )

    try:
        snapshot_dir: str = snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
        )
    except RepositoryNotFoundError as exc:
        logger.error(
            "event=download_failed repo_id=%s error_type=%s",
            repo_id, type(exc).__name__,
        )
        raise DownloadError(
            f"Repository not found on Hugging Face Hub: '{repo_id}'",
            details={"repo_id": repo_id, "error_type": "RepositoryNotFoundError"},
        ) from exc
    except Exception as exc:
        logger.error(
            "event=download_failed repo_id=%s error_type=%s",
            repo_id, type(exc).__name__,
        )
        raise DownloadError(
            f"Failed to download '{repo_id}': {exc}",
            details={
                "repo_id": repo_id,
                "cache_dir": cache_dir,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        ) from exc

    logger.debug("event=snapshot_downloaded snapshot_dir=%s", snapshot_dir)

    # ---- LFS sanity check --------------------------------------------------
    lfs_patterns = _read_lfs_patterns(snapshot_dir)
    if lfs_patterns:
        logger.debug(
            "event=lfs_patterns_found snapshot_dir=%s pattern_count=%d",
            snapshot_dir, len(lfs_patterns),
        )
        _verify_lfs_files(snapshot_dir, lfs_patterns)
    else:
        logger.info(
            "event=checksum_verification_skipped snapshot_dir=%s "
            "reason=no_gitattributes_or_no_lfs_entries",
            snapshot_dir,
        )

    logger.info(
        "event=download_complete repo_id=%s snapshot_dir=%s",
        repo_id, snapshot_dir,
    )
    return snapshot_dir


__all__ = [
    "DownloadError",
    "download_pretrained",
]

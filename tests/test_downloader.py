"""
tests/test_downloader.py

Unit tests for fusion/models/pretrained/downloader.py.

Test cases (per Aman's spec):
  - test_download_pretrained_calls_snapshot_download_with_correct_repo_id
  - test_download_creates_cache_dir_if_missing
  - test_checksum_verification_passes_on_valid_files
  - test_checksum_verification_fails_on_corrupted_file
  - test_download_skips_reverify_if_no_gitattributes_present
"""
from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch, mock_open

import pytest

from fusion.models.pretrained.downloader import (
    DownloadError,
    download_pretrained,
    _is_lfs_pointer,
    _read_lfs_patterns,
    _verify_lfs_files,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_SNAPSHOT_DIR = "/fake/snapshot/dir"
_FAKE_REPO_ID = "org/fake-model"


def _make_snapshot_download_mock(return_value: str = _FAKE_SNAPSHOT_DIR):
    return MagicMock(return_value=return_value)


# ---------------------------------------------------------------------------
# Tests — snapshot_download integration
# ---------------------------------------------------------------------------

class TestSnapshotDownload:

    @patch("fusion.models.pretrained.downloader.snapshot_download",
           create=True)
    def test_download_pretrained_calls_snapshot_download_with_correct_repo_id(
        self, mock_sd
    ):
        """snapshot_download must be called with repo_id= and cache_dir=."""
        mock_sd.return_value = _FAKE_SNAPSHOT_DIR

        with (
            patch("fusion.models.pretrained.downloader._ensure_dir"),
            patch("fusion.models.pretrained.downloader._read_lfs_patterns",
                  return_value=[]),
        ):
            # We patch the huggingface_hub import at module level
            with patch.dict("sys.modules", {
                "huggingface_hub": MagicMock(
                    snapshot_download=mock_sd,
                    utils=MagicMock(
                        RepositoryNotFoundError=Exception,
                        EntryNotFoundError=Exception,
                    ),
                )
            }):
                result = download_pretrained(
                    _FAKE_REPO_ID, cache_dir="/tmp/test_cache"
                )

        assert result == _FAKE_SNAPSHOT_DIR
        mock_sd.assert_called_once_with(
            repo_id=_FAKE_REPO_ID,
            cache_dir="/tmp/test_cache",
        )

    def test_download_creates_cache_dir_if_missing(self):
        """cache_dir is created when it does not exist yet."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_cache = os.path.join(tmpdir, "new", "nested", "cache")
            assert not os.path.exists(new_cache)

            fake_snapshot = os.path.join(tmpdir, "snapshot")
            os.makedirs(fake_snapshot)

            with patch.dict("sys.modules", {
                "huggingface_hub": MagicMock(
                    snapshot_download=MagicMock(return_value=fake_snapshot),
                    utils=MagicMock(
                        RepositoryNotFoundError=Exception,
                        EntryNotFoundError=Exception,
                    ),
                )
            }):
                with patch(
                    "fusion.models.pretrained.downloader._read_lfs_patterns",
                    return_value=[],
                ):
                    download_pretrained(_FAKE_REPO_ID, cache_dir=new_cache)

            assert os.path.isdir(new_cache)


# ---------------------------------------------------------------------------
# Tests — LFS pointer detection
# ---------------------------------------------------------------------------

class TestLFSPointerDetection:

    def test_is_lfs_pointer_returns_true_for_pointer_stub(self, tmp_path):
        """A real LFS pointer stub (magic header + small size) is detected."""
        stub = (
            b"version https://git-lfs.github.com/spec/v1\n"
            b"oid sha256:abc123\nsize 123456789\n"
        )
        pointer_file = tmp_path / "model.bin"
        pointer_file.write_bytes(stub)
        assert _is_lfs_pointer(str(pointer_file)) is True

    def test_is_lfs_pointer_returns_false_for_real_binary(self, tmp_path):
        """A file larger than _LFS_POINTER_MAX_BYTES is not a pointer."""
        real_file = tmp_path / "model.bin"
        real_file.write_bytes(b"\x00" * 1024)   # 1 KB — well above threshold
        assert _is_lfs_pointer(str(real_file)) is False

    def test_is_lfs_pointer_returns_false_for_wrong_header(self, tmp_path):
        """Small file without the LFS magic header is not a pointer."""
        normal_small = tmp_path / "config.json"
        normal_small.write_text('{"hidden_size": 768}')
        assert _is_lfs_pointer(str(normal_small)) is False


class TestReadLFSPatterns:

    def test_returns_empty_list_when_no_gitattributes(self, tmp_path):
        """If .gitattributes is absent, return []."""
        patterns = _read_lfs_patterns(str(tmp_path))
        assert patterns == []

    def test_parses_lfs_lines_correctly(self, tmp_path):
        """Lines with 'filter=lfs' yield the glob pattern from column 0."""
        ga = tmp_path / ".gitattributes"
        ga.write_text(
            "*.bin filter=lfs diff=lfs merge=lfs -text\n"
            "*.safetensors filter=lfs diff=lfs merge=lfs -text\n"
            "*.txt text\n"          # not LFS
        )
        patterns = _read_lfs_patterns(str(tmp_path))
        assert "*.bin" in patterns
        assert "*.safetensors" in patterns
        assert "*.txt" not in patterns


# ---------------------------------------------------------------------------
# Tests — checksum / verification logic
# ---------------------------------------------------------------------------

class TestChecksumVerification:

    def test_checksum_verification_passes_on_valid_files(self, tmp_path):
        """No warning when all LFS-tracked files are real binaries (not stubs)."""
        ga = tmp_path / ".gitattributes"
        ga.write_text("*.bin filter=lfs diff=lfs merge=lfs -text\n")

        real_bin = tmp_path / "model.bin"
        real_bin.write_bytes(b"\x00" * 1024)   # large enough — not a stub

        # _verify_lfs_files should log event=checksum_verified (no warning)
        import logging
        with patch(
            "fusion.models.pretrained.downloader.logger"
        ) as mock_logger:
            _verify_lfs_files(str(tmp_path), ["*.bin"])

        # warning should NOT have been called
        mock_logger.warning.assert_not_called()
        mock_logger.info.assert_called()
        info_calls = " ".join(str(c) for c in mock_logger.info.call_args_list)
        assert "checksum_verified" in info_calls

    def test_checksum_verification_fails_on_corrupted_file(self, tmp_path):
        """Warning is emitted when a matched file is an LFS pointer stub."""
        ga = tmp_path / ".gitattributes"
        ga.write_text("*.bin filter=lfs diff=lfs merge=lfs -text\n")

        stub = (
            b"version https://git-lfs.github.com/spec/v1\n"
            b"oid sha256:deadbeef\nsize 9999999\n"
        )
        stub_file = tmp_path / "pytorch_model.bin"
        stub_file.write_bytes(stub)

        with patch(
            "fusion.models.pretrained.downloader.logger"
        ) as mock_logger:
            _verify_lfs_files(str(tmp_path), ["*.bin"])

        mock_logger.warning.assert_called_once()
        warning_msg = str(mock_logger.warning.call_args)
        assert "checksum_verification_failed" in warning_msg

    def test_download_skips_reverify_if_no_gitattributes_present(self, tmp_path):
        """event=checksum_verification_skipped when .gitattributes is absent."""
        fake_snapshot = tmp_path / "snapshot"
        fake_snapshot.mkdir()
        # No .gitattributes inside snapshot dir

        with patch.dict("sys.modules", {
            "huggingface_hub": MagicMock(
                snapshot_download=MagicMock(return_value=str(fake_snapshot)),
                utils=MagicMock(
                    RepositoryNotFoundError=Exception,
                    EntryNotFoundError=Exception,
                ),
            )
        }):
            cache = str(tmp_path / "cache")
            with patch(
                "fusion.models.pretrained.downloader.logger"
            ) as mock_logger:
                download_pretrained(_FAKE_REPO_ID, cache_dir=cache)

        info_calls = " ".join(str(c) for c in mock_logger.info.call_args_list)
        assert "checksum_verification_skipped" in info_calls


# ---------------------------------------------------------------------------
# Tests — network / repo-not-found errors
# ---------------------------------------------------------------------------

class TestDownloadErrors:

    def test_repo_not_found_raises_download_error(self):
        """RepositoryNotFoundError from hub is re-raised as DownloadError."""
        class FakeRepoNotFound(Exception):
            pass

        with patch.dict("sys.modules", {
            "huggingface_hub": MagicMock(
                snapshot_download=MagicMock(side_effect=FakeRepoNotFound("404")),
                utils=MagicMock(
                    RepositoryNotFoundError=FakeRepoNotFound,
                    EntryNotFoundError=Exception,
                ),
            )
        }):
            with pytest.raises(DownloadError) as exc_info:
                download_pretrained("nonexistent/model", cache_dir="/tmp/c")

        assert "nonexistent/model" in str(exc_info.value)

    def test_generic_network_error_raises_download_error(self):
        """Any unexpected exception from snapshot_download is wrapped."""
        with patch.dict("sys.modules", {
            "huggingface_hub": MagicMock(
                snapshot_download=MagicMock(
                    side_effect=ConnectionError("network unreachable")
                ),
                utils=MagicMock(
                    RepositoryNotFoundError=Exception,
                    EntryNotFoundError=Exception,
                ),
            )
        }):
            with pytest.raises(DownloadError):
                download_pretrained(_FAKE_REPO_ID, cache_dir="/tmp/c")

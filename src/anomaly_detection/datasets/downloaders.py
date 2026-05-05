"""Download adapters for dataset sources."""

import hashlib
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

from ..config import DatasetSettings
from .catalog import DatasetSpec

RAW_FILE_SUFFIX = ".raw"
"""Raw dataset artifact suffix."""


class DownloadError(RuntimeError):
    """Raised when dataset download fails."""


@dataclass(frozen=True, slots=True)
class DownloadResult:
    """Normalized download output for pipeline provenance.

    Attributes:
        path: Final downloaded artifact path.
        backend_used: Backend that produced output.
        sha256: Hex digest of final artifact bytes.
    """

    path: Path
    backend_used: str
    sha256: str


class DatasetDownloader:
    """Downloader for URL-backed dataset sources."""

    def __init__(self, settings: DatasetSettings):
        """Create downloader with environment-configurable backend commands."""
        self.settings = settings

    def download(self, spec: DatasetSpec, target_dir: Path) -> DownloadResult:
        """Download raw dataset if possible.

        Args:
            spec: Dataset specification.
            target_dir: Destination directory for raw artifacts.

        Returns:
            Download result with provenance fields.
        """
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{spec.dataset_id.replace('/', '__')}{RAW_FILE_SUFFIX}"
        if target.exists():
            text_prefix = target.read_text(encoding="utf-8", errors="ignore")[:256]
            if not zipfile.is_zipfile(target) and not _looks_like_csv_text(text_prefix):
                raise DownloadError(
                    f"Cached artifact format unsupported for {spec.source_ref}: {target}"
                )
            return DownloadResult(
                path=target,
                backend_used="cache",
                sha256=_sha256(target),
            )
        return self._download_http(spec, target)

    def _download_http(self, spec: DatasetSpec, target: Path) -> DownloadResult:
        """Fallback HTTP downloader.

        Args:
            spec: Dataset specification using URL in `source_ref`.
            target: Destination file.

        Returns:
            Path to downloaded file.

        Raises:
            DownloadError: If source is not a valid URL.
        """
        if spec.source_type != "requests":
            raise DownloadError(f"Unsupported source_type={spec.source_type} for {spec.dataset_id}")
        if not spec.source_ref.startswith(("http://", "https://")):
            raise DownloadError(f"Cannot fetch non-url source_ref={spec.source_ref}")
        try:
            with urlopen(  # noqa: S310
                spec.source_ref, timeout=self.settings.download_timeout_seconds
            ) as response:
                target.write_bytes(response.read())
        except Exception as exc:
            raise DownloadError(f"HTTP download failed for {spec.source_ref}") from exc
        return DownloadResult(
            path=target,
            backend_used="requests",
            sha256=_sha256(target),
        )


def _sha256(path: Path) -> str:
    """Compute SHA256 for file."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _looks_like_csv_text(text_prefix: str) -> bool:
    """Detect probable CSV payload in textual cache."""
    first_line = text_prefix.splitlines()[0] if text_prefix.splitlines() else ""
    return "," in first_line and len(first_line.strip()) > 0

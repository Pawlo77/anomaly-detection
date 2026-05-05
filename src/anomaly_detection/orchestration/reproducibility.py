"""Environment, repository provenance, and hardware snapshots for reproducibility.

Collected into ``ReproducibilityReport`` for MLflow uploads or informational logging
when remote tracking backends are intentionally disabled.
"""

import importlib
import importlib.metadata
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ReproducibilityReport:
    """Minimal environment snapshot stored alongside each MLflow run.

    Attributes:
        processor: Processor architecture string.
        system: Operating system name.
        release: Operating system release string.
        machine: Machine architecture string.
        python_version: Python interpreter version.
        python_implementation: Python implementation name.
        platform: Full platform string.
        total_ram_bytes: Total physical RAM in bytes when detectable.
        total_disk_bytes: Total filesystem capacity for repository volume in bytes.
        installed_packages: Installed package version mapping.
        git_commit_hash: Current git commit hash.
        git_branch: Current git branch or ref name.
        git_dirty: Whether the working tree has uncommitted changes.
        git_status_porcelain: Raw porcelain status for the working tree.
        created_at: ISO-8601 timestamp when report was created.
    """

    processor: str
    system: str
    release: str
    machine: str
    python_version: str
    python_implementation: str
    platform: str
    total_ram_bytes: int
    total_disk_bytes: int
    installed_packages: dict[str, str]
    git_commit_hash: str
    git_branch: str
    git_dirty: bool
    git_status_porcelain: str
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report to JSON-friendly values."""
        return {
            "processor": self.processor,
            "system": self.system,
            "release": self.release,
            "machine": self.machine,
            "python_version": self.python_version,
            "python_implementation": self.python_implementation,
            "platform": self.platform,
            "total_ram_bytes": self.total_ram_bytes,
            "total_disk_bytes": self.total_disk_bytes,
            "installed_packages": self.installed_packages,
            "git_commit_hash": self.git_commit_hash,
            "git_branch": self.git_branch,
            "git_dirty": self.git_dirty,
            "git_status_porcelain": self.git_status_porcelain,
            "created_at": self.created_at,
        }


def collect_reproducibility_report() -> ReproducibilityReport:
    """Capture processor, OS, RAM, disk, pip packages, and git provenance."""
    repo_root = Path(__file__).resolve().parents[3]
    git_commit_hash = _read_git_commit_hash(repo_root)
    git_branch = _read_git_branch(repo_root)
    git_status_porcelain = _read_git_status_porcelain(repo_root)
    total_ram_bytes = _read_total_ram_bytes()
    total_disk_bytes = _read_total_disk_bytes(repo_root)

    return ReproducibilityReport(
        processor=platform.processor() or "unknown",
        system=platform.system(),
        release=platform.release(),
        machine=platform.machine(),
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        platform=platform.platform(),
        total_ram_bytes=total_ram_bytes,
        total_disk_bytes=total_disk_bytes,
        installed_packages=_cached_installed_packages(),
        git_commit_hash=git_commit_hash,
        git_branch=git_branch,
        git_dirty=bool(git_status_porcelain.strip()),
        git_status_porcelain=git_status_porcelain,
    )


def _read_git_commit_hash(repo_root: Path) -> str:
    """Resolve the current git commit hash from repository metadata."""
    git_dir = repo_root / ".git"
    head_path = git_dir / "HEAD"
    if git_dir.is_file():
        git_dir_text = git_dir.read_text(encoding="utf-8").strip()
        if git_dir_text.startswith("gitdir:"):
            git_dir = (repo_root / git_dir_text.split("gitdir:", maxsplit=1)[1].strip()).resolve()
            head_path = git_dir / "HEAD"

    try:
        head_contents = head_path.read_text(encoding="utf-8").strip()
    except OSError:
        return "unknown"

    if head_contents.startswith("ref: "):
        ref_name = head_contents.removeprefix("ref: ").strip()
        ref_path = git_dir / ref_name
        try:
            return ref_path.read_text(encoding="utf-8").strip() or "unknown"
        except OSError:
            return "unknown"

    return head_contents or "unknown"


def _read_git_branch(repo_root: Path) -> str:
    """Resolve the checked-out branch name or a detached-HEAD marker."""
    git_dir = repo_root / ".git"
    head_path = git_dir / "HEAD"
    if git_dir.is_file():
        git_dir_text = git_dir.read_text(encoding="utf-8").strip()
        if git_dir_text.startswith("gitdir:"):
            git_dir = (repo_root / git_dir_text.split("gitdir:", maxsplit=1)[1].strip()).resolve()
            head_path = git_dir / "HEAD"

    try:
        head_contents = head_path.read_text(encoding="utf-8").strip()
    except OSError:
        return "unknown"

    if head_contents.startswith("ref: "):
        return head_contents.removeprefix("ref: ").strip() or "unknown"
    return "detached-head"


def _read_git_status_porcelain(repo_root: Path) -> str:
    """Return ``git status --porcelain`` output for dirty-tree detection."""
    try:
        git_executable = shutil.which("git")
        if git_executable is None:
            return ""
        completed = subprocess.run(  # noqa: S603
            [git_executable, "status", "--porcelain=v1", "-uall"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return completed.stdout.strip()


def _read_total_ram_bytes() -> int:
    """Best-effort total physical memory in bytes."""
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if isinstance(pages, int) and isinstance(page_size, int) and pages > 0 and page_size > 0:
            return int(pages * page_size)
    except (AttributeError, OSError, ValueError):
        pass

    if platform.system() == "Darwin":
        try:
            sysctl_executable = shutil.which("sysctl")
            if sysctl_executable is None:
                return 0
            completed = subprocess.run(  # noqa: S603
                [sysctl_executable, "-n", "hw.memsize"],
                check=True,
                capture_output=True,
                text=True,
            )
            value = int(completed.stdout.strip())
            if value > 0:
                return value
        except (OSError, ValueError, subprocess.CalledProcessError):
            pass

    return 0


def _read_total_disk_bytes(repo_root: Path) -> int:
    """Total capacity of the filesystem hosting ``repo_root``."""
    try:
        return int(shutil.disk_usage(repo_root).total)
    except OSError:
        return 0


def _collect_installed_packages() -> dict[str, str]:
    """Map distribution names to versions for the active interpreter."""
    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        version = distribution.version
        if not name or not version:
            continue
        packages[name] = version
    return dict(sorted(packages.items(), key=lambda item: item[0].lower()))


@lru_cache(maxsize=1)
def _cached_installed_packages() -> dict[str, str]:
    """Memoize the package scan for repeated calls in one process."""
    return _collect_installed_packages()

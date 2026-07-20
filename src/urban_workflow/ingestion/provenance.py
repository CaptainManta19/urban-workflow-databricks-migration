import hashlib
from datetime import datetime, timezone
from pathlib import Path

from urban_workflow.ingestion.schemas import SourceProvenance


def to_project_relative(path: Path, project_root: Path) -> str:
    resolved_path = path.resolve()

    try:
        return str(resolved_path.relative_to(project_root.resolve()))
    except ValueError:
        return str(resolved_path)


def compute_sha256(file_path: Path) -> str:
    digest = hashlib.sha256()

    with file_path.open("rb") as file_handle:
        while chunk := file_handle.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def is_git_lfs_pointer_file(file_path: Path) -> bool:
    try:
        with file_path.open("rb") as file_handle:
            header = file_handle.read(256)
    except OSError:
        return False

    return header.startswith(b"version https://git-lfs.github.com/spec/v1\n")


def ensure_file_is_materialised(file_path: Path) -> None:
    if is_git_lfs_pointer_file(file_path):
        raise ValueError(
            "File is an unresolved Git LFS pointer. Install Git LFS, then run "
            "'git lfs pull' after cloning the repository. GitHub ZIP downloads "
            "do not fetch required LFS objects."
        )


def build_provenance(file_path: Path, project_root: Path) -> SourceProvenance:
    stat_result = file_path.stat()

    return SourceProvenance(
        filename=file_path.name,
        relative_path=to_project_relative(file_path, project_root),
        absolute_path=str(file_path.resolve()),
        file_size_bytes=stat_result.st_size,
        modified_at=datetime.fromtimestamp(stat_result.st_mtime, tz=timezone.utc),
        sha256=compute_sha256(file_path),
    )
"""Capped on-device store for snapshots, served under /media like the lab's Tapo gateway."""

import re
import threading
from datetime import datetime, timezone
from pathlib import Path

from .models import MediaFile

LENS = "main"
_NAME = re.compile(r"^capture_\d{8}T\d{6}_\d{6}Z\.jpg$")


class MediaStore:
    def __init__(self, root, max_files=200):
        self.root = Path(root).expanduser()
        self.dir = self.root / "snapshots" / LENS
        self.dir.mkdir(parents=True, exist_ok=True)
        self.max_files = max(1, int(max_files))
        self._lock = threading.Lock()

    @staticmethod
    def url_for(name):
        return f"/media/snapshots/{LENS}/{name}"

    @staticmethod
    def rel_path(name):
        return f"snapshots/{LENS}/{name}"

    def save(self, data, taken_at):
        name = f"capture_{taken_at.strftime('%Y%m%dT%H%M%S_%f')}Z.jpg"
        with self._lock:
            tmp = self.dir / (name + ".part")
            tmp.write_bytes(data)
            tmp.replace(self.dir / name)
            files = self._files()
            for old in files[: max(0, len(files) - self.max_files)]:
                old.unlink(missing_ok=True)
        return name

    def _files(self):
        return sorted(p for p in self.dir.iterdir() if _NAME.match(p.name))

    def count(self):
        return len(self._files())

    def listing(self):
        items = []
        for path in reversed(self._files()):
            stat = path.stat()
            items.append(MediaFile(
                name=path.name, lens=LENS, kind="snapshots", bytes=stat.st_size,
                mtime=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                url=self.url_for(path.name), abs_path=self.rel_path(path.name),
            ))
        return items

    def resolve(self, kind, lens, name):
        """Return the file path for a media URL, or None if it is not a stored snapshot."""
        if kind != "snapshots" or lens != LENS or not _NAME.match(name):
            return None
        path = self.dir / name
        return path if path.is_file() else None

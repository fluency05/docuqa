"""Document loading from files and directories."""

from __future__ import annotations

from pathlib import Path

from .types import Document

SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".rst", ".pdf"}


class UnsupportedFileTypeError(ValueError):
    """Raised when a file has an extension docuqa cannot process."""


class DocumentLoader:
    """Load text documents from a single file or a directory tree."""

    def load_path(self, path: str | Path) -> list[Document]:
        """Load one file, or every supported file under a directory (recursively)."""
        target = Path(path)
        if target.is_dir():
            documents: list[Document] = []
            for extension in sorted(SUPPORTED_EXTENSIONS):
                for file in sorted(target.rglob(f"*{extension}")):
                    documents.append(self._load_file(file))
            return documents
        return [self._load_file(target)]

    def _load_file(self, path: Path) -> Document:
        extension = path.suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFileTypeError(f"Unsupported file type: {extension!r}")
        if extension == ".pdf":
            text = self._read_pdf(path)
        else:
            text = path.read_text(encoding="utf-8")
        return Document(source=str(path), content=text)

    @staticmethod
    def _read_pdf(path: Path) -> str:
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "PDF support requires the 'pypdf' package. Install it with: pip install pypdf"
            ) from exc
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)

"""Document loading from files, directories, and raw bytes."""

from __future__ import annotations

import io
from pathlib import Path

from .types import Document

SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".rst", ".pdf"}


class UnsupportedFileTypeError(ValueError):
    """Raised when a file has an extension docuqa cannot process."""


class DocumentLoader:
    """Load text documents from a file, a directory tree, or raw bytes."""

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

    def load_bytes(self, content: bytes, filename: str) -> Document:
        """Load a document from raw bytes (e.g. a browser upload)."""
        extension = Path(filename).suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFileTypeError(f"Unsupported file type: {extension!r}")
        if extension == ".pdf":
            text = self._read_pdf_bytes(content)
        else:
            text = self._decode_text(content)
        return Document(source=filename, content=text)

    def _load_file(self, path: Path) -> Document:
        extension = path.suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFileTypeError(f"Unsupported file type: {extension!r}")
        if extension == ".pdf":
            text = self._read_pdf_bytes(path.read_bytes())
        else:
            text = self._decode_text(path.read_bytes())
        return Document(source=str(path), content=text)

    @staticmethod
    def _decode_text(content: bytes) -> str:
        """Decode text, trying UTF-8, then GBK, then Latin-1."""
        for encoding in ("utf-8", "gbk", "latin-1"):
            try:
                return content.decode(encoding)
            except (UnicodeDecodeError, LookupError):
                continue
        return content.decode("utf-8", errors="replace")

    @staticmethod
    def _read_pdf_bytes(content: bytes) -> str:
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "PDF support requires the 'pypdf' package. Install it with: pip install pypdf"
            ) from exc
        reader = PdfReader(io.BytesIO(content))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)

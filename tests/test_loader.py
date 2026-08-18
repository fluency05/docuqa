import pytest

from docuqa.loader import DocumentLoader, UnsupportedFileTypeError


def test_loads_text_file(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("hello", encoding="utf-8")
    documents = DocumentLoader().load_path(path)
    assert len(documents) == 1
    assert documents[0].content == "hello"
    assert documents[0].source == str(path)


def test_loads_directory_recursively(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.md").write_text("one", encoding="utf-8")
    (tmp_path / "sub" / "b.txt").write_text("two", encoding="utf-8")
    documents = DocumentLoader().load_path(tmp_path)
    assert {d.content for d in documents} == {"one", "two"}


def test_skips_unsupported_extensions_in_directory(tmp_path):
    (tmp_path / "a.txt").write_text("ok", encoding="utf-8")
    (tmp_path / "ignored.bin").write_bytes(b"\x00\x01")
    documents = DocumentLoader().load_path(tmp_path)
    assert [d.content for d in documents] == ["ok"]


def test_unsupported_file_raises(tmp_path):
    path = tmp_path / "a.bin"
    path.write_bytes(b"\x00")
    with pytest.raises(UnsupportedFileTypeError):
        DocumentLoader().load_path(path)

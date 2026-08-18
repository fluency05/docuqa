from docuqa.chunker import TextChunker
from docuqa.types import Document


def test_short_text_is_a_single_chunk():
    chunker = TextChunker(chunk_size=100, overlap=0)
    doc = Document(source="t.txt", content="hello world")
    chunks = chunker.chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].text == "hello world"


def test_chunks_stay_within_size_without_overlap():
    chunker = TextChunker(chunk_size=50, overlap=0)
    doc = Document(source="t.txt", content=("word " * 200).strip())
    chunks = chunker.chunk(doc)
    assert len(chunks) > 1
    assert all(len(c.text) <= 50 for c in chunks)


def test_overlap_carries_context_between_chunks():
    chunker = TextChunker(chunk_size=40, overlap=10)
    doc = Document(source="t.txt", content=("abcde " * 40).strip())
    chunks = chunker.chunk(doc)
    assert len(chunks) > 1
    assert chunks[1].text.startswith(chunks[0].text[-10:])


def test_chunk_ids_and_sources_are_preserved():
    chunker = TextChunker(chunk_size=100, overlap=0)
    doc = Document(source="notes.md", content="hello world")
    chunks = chunker.chunk(doc)
    assert chunks[0].id == "notes.md#0"
    assert chunks[0].source == "notes.md"


def test_empty_document_produces_no_chunks():
    chunker = TextChunker()
    assert chunker.chunk(Document(source="t.txt", content="   \n\n  ")) == []


def test_invalid_parameters_raise():
    try:
        TextChunker(chunk_size=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for non-positive chunk_size")
    try:
        TextChunker(chunk_size=100, overlap=100)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for overlap >= chunk_size")


def test_chinese_long_text_splits_at_sentence_boundaries():
    chunker = TextChunker(chunk_size=80, overlap=0)
    sentence = "这是第一个句子。这是第二个句子！这是第三个句子？"
    long_text = sentence * 10

    chunks = chunker.chunk(Document(source="zh.txt", content=long_text))

    assert len(chunks) > 1
    for chunk in chunks[:-1]:
        assert chunk.text[-1] in "。！？"

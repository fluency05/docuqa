from docuqa.store import InMemoryVectorStore
from docuqa.types import Chunk


def _chunk(index: int, text: str) -> Chunk:
    return Chunk(id=f"c{index}", text=text, source="s.txt", index=index)


def test_search_returns_most_similar_chunk():
    store = InMemoryVectorStore()
    store.add([_chunk(0, "apple banana")], [[1.0, 0.0]])
    store.add([_chunk(1, "car dog")], [[0.0, 1.0]])
    results = store.search([1.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0][0].text == "apple banana"


def test_search_on_empty_store_returns_empty():
    store = InMemoryVectorStore()
    assert store.search([1.0, 0.0], top_k=3) == []


def test_search_respects_top_k():
    store = InMemoryVectorStore()
    store.add(
        [_chunk(i, f"chunk {i}") for i in range(5)],
        [[float(i), 1.0] for i in range(5)],
    )
    assert len(store.search([4.0, 1.0], top_k=3)) == 3
    assert len(store) == 5


def test_save_and_load_roundtrip(tmp_path):
    store = InMemoryVectorStore()
    store.add([_chunk(0, "hello world")], [[1.0, 2.0]])
    store.save(tmp_path)

    loaded = InMemoryVectorStore.load(tmp_path)
    assert len(loaded) == 1
    results = loaded.search([1.0, 2.0], top_k=1)
    assert results[0][0].text == "hello world"


def test_load_missing_index_is_empty(tmp_path):
    store = InMemoryVectorStore.load(tmp_path)
    assert len(store) == 0
    assert store.search([1.0, 0.0], top_k=3) == []

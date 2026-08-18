from docuqa.embedder import HashingEmbedder


def test_hashing_embedder_tokenizes_cjk_bigrams():
    emb = HashingEmbedder()
    tokens = emb._tokens("星辰笔记定价")
    assert "星辰" in tokens
    assert "定价" in tokens
    assert len(tokens) >= 4


def test_hashing_embedder_tokenizes_english_words():
    emb = HashingEmbedder()
    tokens = emb._tokens("Hello, world!")
    assert "hello" in tokens
    assert "world" in tokens


def test_hashing_embedder_vectors_have_unit_norm():
    emb = HashingEmbedder()
    vector = emb.embed_query("星辰笔记的定价是多少")
    norm = sum(v * v for v in vector) ** 0.5
    assert abs(norm - 1.0) < 1e-6

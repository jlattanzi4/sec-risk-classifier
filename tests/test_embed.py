import numpy as np

from pipeline.embed import Embedder


class Stub:
    """Embeds a text as (its length, 1), and records the order it was asked in."""

    def __init__(self):
        self.seen = []

    def encode(self, texts, batch_size):
        self.seen.extend(texts)
        return np.array([[len(t), 1.0] for t in texts], dtype=np.float64)


def test_rows_come_back_in_input_order_whatever_the_lengths():
    stub = Stub()
    texts = ["ccc", "a", "eeeee", "bb"]
    out = Embedder(stub, "stub").encode(texts)
    assert stub.seen == ["eeeee", "ccc", "bb", "a"]  # longest first, so batches pad little
    expected = np.array([[len(t), 1.0] for t in texts])
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    np.testing.assert_allclose(out, expected, rtol=1e-6)


def test_rows_are_unit_length_float32():
    out = Embedder(Stub(), "stub").encode(["some text", "more"])
    assert out.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, rtol=1e-6)


def test_no_texts_gives_an_empty_matrix_of_the_right_width():
    out = Embedder(Stub(), "stub", dim=2).encode([])
    assert out.shape == (0, 2) and out.dtype == np.float32

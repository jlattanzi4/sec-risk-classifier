"""Sentence embeddings, for telling a reworded sentence from a new one.

`MODEL` is the model the pipeline uses. It was chosen by test, on the half of
the known-answer set kept for tuning (`scripts/compare_embedders.py`), not by
default.
"""

import numpy as np

MODEL = "BAAI/bge-small-en-v1.5"
CANDIDATES = (
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
    "minishlab/potion-base-8M",     # static: word vectors averaged, no attention
)


class Embedder:
    """A model that turns sentences into unit vectors."""

    def __init__(self, model, name: str, dim: int | None = None, max_tokens: int | None = None):
        self.model, self.name, self.dim, self.max_tokens = model, name, dim, max_tokens

    def encode(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        """One L2-normalized `float32` row per text, in the order given.

        Texts are embedded longest first, so each batch holds sentences of
        similar length and little of it is padding.
        """
        if not texts:
            return np.zeros((0, self.dim or 0), dtype=np.float32)
        order = np.argsort([-len(t) for t in texts], kind="stable")
        vectors = np.asarray(self.model.encode([texts[i] for i in order], batch_size=batch_size))
        vectors = vectors.astype(np.float32)
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        out = np.empty_like(vectors)
        out[order] = vectors
        return out

    def over_limit(self, texts: list[str]) -> float:
        """Share of texts longer than the model can read (they would be cut off)."""
        if self.max_tokens is None:
            return 0.0
        lengths = self.model.tokenizer(texts, add_special_tokens=True, truncation=False)
        return float(np.mean([len(ids) > self.max_tokens for ids in lengths["input_ids"]]))


class _SentenceTransformer:
    def __init__(self, name: str):
        import torch
        from sentence_transformers import SentenceTransformer

        on_gpu = torch.backends.mps.is_available()
        self.inner = SentenceTransformer(name, device="mps" if on_gpu else "cpu")
        if on_gpu:
            # Half precision is about 1.3 times faster here and moves a cosine by
            # under 0.001, less than the float16 the vectors are stored in.
            self.inner.half()
        self.tokenizer = self.inner.tokenizer

    def encode(self, texts, batch_size):
        return self.inner.encode(
            texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False
        )


class _Static:
    def __init__(self, name: str):
        from model2vec import StaticModel

        self.inner = StaticModel.from_pretrained(name)

    def encode(self, texts, batch_size):
        return self.inner.encode(texts)


def load_embedder(name: str = MODEL) -> Embedder:
    """Load a candidate by its Hugging Face name. Uses the Apple GPU when there is one."""
    if name.startswith("minishlab/"):
        model = _Static(name)
        return Embedder(model, name, dim=model.inner.dim, max_tokens=None)
    model = _SentenceTransformer(name)
    dim = model.inner.get_embedding_dimension()
    return Embedder(model, name, dim=dim, max_tokens=model.inner.max_seq_length)

#!/usr/bin/env python
# coding: utf-8
"""Swappable semantic-similarity scorers for the attacker's similarity reward.

This mirrors the existing swappable-judge pattern (``llama_guard_judge`` /
``openai_moderation_judge``): pick an implementation by name with
``build_scorer(name)`` and call ``scorer.score(src_docs, adv_docs)`` to get one
cosine-similarity value per ``(source, adversarial)`` document pair.

Why this module exists
----------------------
The similarity reward used to be an inline ``getUSEcosSimilarity(...)`` function
duplicated across the trainer and the evaluators, with the Universal Sentence
Encoder (USE) hard-wired in. To run ablations that swap USE for another
embedding backend (anything that turns text into a vector, then takes cosine),
the trainer/eval now go through this single seam.

Parity guarantee
-----------------
``USESimilarityScorer`` delegates to :func:`getUSEcosSimilarity`, which is the
*verbatim* original computation. So with the default ``--sim_scorer use`` the
reward values are numerically identical to before this refactor.
"""
from typing import Any, Callable, List, Optional, Sequence

import torch

# The exact TF-Hub URL the trainer/eval used inline before the refactor.
DEFAULT_USE_URL = (
    "https://kaggle.com/models/google/universal-sentence-encoder/"
    "TensorFlow2/universal-sentence-encoder/1"
)


def getUSEcosSimilarity(srcDocs: List[str], copyDocs: List[str], embed: Any) -> List[float]:
    """Cosine similarity per (src, copy) pair using a USE-style ``embed`` model.

    Kept byte-for-byte identical to the original inline implementation so that
    results stay comparable and other callers (e.g. the verbose evaluator) keep
    working. ``embed`` is a loaded TF-Hub USE model; ``embed([a, b])["outputs"]``
    yields the two embeddings.
    """
    USEcosinSimilarity = []
    sim_metric = torch.nn.CosineSimilarity(dim=1)
    for src, copy in zip(srcDocs, copyDocs):
        emb1, emb2 = embed([src, copy])["outputs"]
        emb1, emb2 = torch.tensor(emb1.numpy()), torch.tensor(emb2.numpy())
        srcEmb = torch.unsqueeze(emb1, dim=0)  # [embSz] -> [1, embSz]
        advEmb = torch.unsqueeze(emb2, dim=0)
        es = sim_metric(srcEmb, advEmb)
        USEcosinSimilarity.append(es.item())
    return USEcosinSimilarity


class SimilarityScorer:
    """Interface: return one cosine similarity in [-1, 1] per (src, adv) pair."""

    def score(self, src_docs: List[str], adv_docs: List[str]) -> List[float]:
        raise NotImplementedError


class USESimilarityScorer(SimilarityScorer):
    """Universal Sentence Encoder (TF-Hub). The default, parity-preserving scorer.

    Pass an already-loaded ``embed`` model to reuse one, or leave it ``None`` to
    lazily ``hub.load`` the default USE model (import of tensorflow_hub is
    deferred so this module is importable without TF installed).
    """

    def __init__(self, embed: Any = None, url: str = DEFAULT_USE_URL):
        if embed is None:
            import tensorflow_hub as hub  # deferred: only needed for USE
            embed = hub.load(url)
        self.embed = embed

    def score(self, src_docs: List[str], adv_docs: List[str]) -> List[float]:
        return getUSEcosSimilarity(src_docs, adv_docs, self.embed)


class EmbeddingAPISimilarityScorer(SimilarityScorer):
    """Reference adapter for ANY embedding backend (the ablation target).

    Provide ``embed_fn(texts) -> list-of-vectors`` wrapping whatever produces an
    embedding per text -- an OpenAI/Cohere REST call, a local
    sentence-transformers model, an HTTP endpoint, etc. This adapter:

    * batches the whole step's texts (source + adversarial) into ``embed_fn``
      calls of at most ``batch_size`` instead of the old one-pair-at-a-time loop;
    * de-duplicates identical texts within a step;
    * optionally memoises embeddings in a diskcache-like ``cache`` keyed by the
      text (so repeated documents are not re-embedded across steps);
    * computes the same ``torch.nn.CosineSimilarity(dim=1)`` per pair as USE.

    Fill in ``embed_fn`` for your provider; nothing else needs to change.
    """

    def __init__(
        self,
        embed_fn: Callable[[List[str]], Sequence[Sequence[float]]],
        cache: Optional[Any] = None,
        batch_size: int = 64,
    ):
        self.embed_fn = embed_fn
        self.cache = cache
        self.batch_size = batch_size
        self._cos = torch.nn.CosineSimilarity(dim=1)

    def _embed_all(self, texts: List[str]) -> dict:
        """Return {text: embedding} for the unique texts, using cache + batching."""
        unique = list(dict.fromkeys(texts))  # preserve order, drop duplicates
        embeddings: dict = {}

        to_compute = []
        for t in unique:
            if self.cache is not None and t in self.cache:
                embeddings[t] = self.cache[t]
            else:
                to_compute.append(t)

        for i in range(0, len(to_compute), self.batch_size):
            batch = to_compute[i : i + self.batch_size]
            vecs = self.embed_fn(batch)
            for t, v in zip(batch, vecs):
                embeddings[t] = v
                if self.cache is not None:
                    self.cache[t] = v
        return embeddings

    def score(self, src_docs: List[str], adv_docs: List[str]) -> List[float]:
        embeddings = self._embed_all(list(src_docs) + list(adv_docs))
        sims = []
        for src, adv in zip(src_docs, adv_docs):
            a = torch.tensor(embeddings[src], dtype=torch.float).unsqueeze(0)
            b = torch.tensor(embeddings[adv], dtype=torch.float).unsqueeze(0)
            sims.append(self._cos(a, b).item())
        return sims


def build_scorer(name: str = "use", **kwargs) -> SimilarityScorer:
    """Factory: ``'use'`` (default, parity) or ``'embedding_api'`` (ablation)."""
    name = (name or "use").lower()
    if name == "use":
        return USESimilarityScorer(**kwargs)
    if name in ("embedding_api", "api", "embedding"):
        return EmbeddingAPISimilarityScorer(**kwargs)
    raise ValueError(
        f"Unknown sim_scorer '{name}'. Available: 'use', 'embedding_api'."
    )

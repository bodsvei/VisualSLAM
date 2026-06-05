"""
vocabulary.py
-------------
Visual vocabulary built from ORB descriptors using hierarchical k-means.

Structure
---------
  Level 0 (root)
    └── k children  (level 1)
          └── k children each  (level 2 = leaf nodes = visual words)

Each leaf = one visual word.
Total words = k^L  (k=10, L=6 → 1,000,000 words like ORB-SLAM)
For speed: k=10, L=3 → 1,000 words  (good for real-time building)

TF-IDF weighting
----------------
  IDF(word_i) = log(N / n_i)
  where N = total documents, n_i = docs containing word_i
  TF  = frequency of word_i in current descriptor set
  Score stored as TF-IDF weighted sparse vector.
"""

from __future__ import annotations
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class BowVector:
    """Sparse TF-IDF weighted bag-of-words vector."""
    word_ids : np.ndarray    # (M,) word indices that appeared
    weights  : np.ndarray    # (M,) TF-IDF weights
    kf_id    : int = -1

    def to_dict(self) -> Dict[int, float]:
        return dict(zip(self.word_ids.tolist(), self.weights.tolist()))

    def l1_score(self, other: "BowVector") -> float:
        """
        L1 score between two BoW vectors (ORB-SLAM formula).
        Score = 1 - 0.5 * |v1 - v2|_L1
        Range [0, 1], higher = more similar.
        """
        d1 = self.to_dict()
        d2 = other.to_dict()
        all_words = set(d1) | set(d2)
        l1 = sum(abs(d1.get(w, 0.0) - d2.get(w, 0.0)) for w in all_words)
        return 1.0 - 0.5 * l1


class VisualVocabulary:
    """
    Hierarchical k-means visual vocabulary.

    Build from scratch on first run (takes ~10s for 1000 words),
    or load a pre-saved vocabulary from disk.

    Parameters
    ----------
    k      : branching factor (children per node)
    levels : depth of tree  →  vocab_size = k^levels
    """

    def __init__(self, k: int = 10, levels: int = 3):
        self.k          = k
        self.levels     = levels
        self.vocab_size = k ** levels
        self._centers   : Optional[np.ndarray] = None    # (vocab_size, 32) uint8
        self._idf       : Optional[np.ndarray] = None    # (vocab_size,) float
        self._n_docs    : int = 0
        self._doc_freq  : np.ndarray = np.zeros(1)       # grows with vocab
        self.is_built   : bool = False

    # ------------------------------------------------------------------ #
    #  Build                                                               #
    # ------------------------------------------------------------------ #

    def build(self, all_descriptors: np.ndarray, verbose: bool = True):
        """
        Build vocabulary from a pool of ORB descriptors.

        Parameters
        ----------
        all_descriptors : (N, 32) uint8 — concatenated from many frames
        """
        from sklearn.cluster import MiniBatchKMeans

        N = len(all_descriptors)
        if verbose:
            print(f"[Vocab] Building vocabulary: {self.vocab_size} words "
                  f"from {N} descriptors...")

        # Convert binary descriptors to float for k-means
        desc_f = all_descriptors.astype(np.float32)

        kmeans = MiniBatchKMeans(
            n_clusters   = self.vocab_size,
            batch_size   = min(4096, N),
            max_iter     = 100,
            random_state = 42,
            n_init       = 3,
            verbose      = 0,
        )
        kmeans.fit(desc_f)

        self._centers  = kmeans.cluster_centers_.astype(np.float32)   # (V, 32)
        self._idf      = np.ones(self.vocab_size, dtype=np.float64)   # uniform until docs added
        self._doc_freq = np.zeros(self.vocab_size, dtype=np.int32)
        self._n_docs   = 0
        self.is_built  = True

        if verbose:
            print(f"[Vocab] Built. Inertia={kmeans.inertia_:.1f}")

    # ------------------------------------------------------------------ #
    #  Transform descriptors → BoW vector                                 #
    # ------------------------------------------------------------------ #

    def transform(
        self,
        descriptors : np.ndarray,    # (N, 32) uint8
        kf_id       : int = -1,
    ) -> BowVector:
        """
        Convert a set of descriptors to a TF-IDF BoW vector.
        Call update_idf() after adding documents to the database
        for IDF weights to take effect.
        """
        assert self.is_built, "Call build() or load() first"

        desc_f  = descriptors.astype(np.float32)
        # Assign each descriptor to nearest visual word (L2 on float features)
        diffs   = desc_f[:, None, :] - self._centers[None, :, :]    # (N, V, 32)
        dists   = np.sum(diffs ** 2, axis=2)                        # (N, V)
        word_assignments = np.argmin(dists, axis=1)                  # (N,)

        # TF = normalised word frequency
        counts  = np.bincount(word_assignments, minlength=self.vocab_size).astype(np.float64)
        tf      = counts / (counts.sum() + 1e-12)

        # TF-IDF
        tfidf   = tf * self._idf

        # Keep only non-zero words (sparse)
        nonzero = np.where(tfidf > 0)[0]

        # L1 normalise
        norm    = tfidf[nonzero].sum()
        weights = tfidf[nonzero] / (norm + 1e-12)

        return BowVector(word_ids=nonzero.astype(np.int32),
                         weights=weights.astype(np.float32),
                         kf_id=kf_id)

    # ------------------------------------------------------------------ #
    #  IDF update                                                          #
    # ------------------------------------------------------------------ #

    def add_document_words(self, bow: BowVector):
        """Update document frequency counts. Call once per keyframe added."""
        self._n_docs += 1
        self._doc_freq[bow.word_ids] += 1

    def update_idf(self):
        """Recompute IDF weights from current document frequencies."""
        n  = max(self._n_docs, 1)
        df = np.maximum(self._doc_freq, 1)
        self._idf = np.log(n / df).astype(np.float64)

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save(self, path: str):
        data = {
            "k"         : self.k,
            "levels"    : self.levels,
            "centers"   : self._centers,
            "idf"       : self._idf,
            "doc_freq"  : self._doc_freq,
            "n_docs"    : self._n_docs,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=4)
        print(f"[Vocab] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "VisualVocabulary":
        with open(path, "rb") as f:
            data = pickle.load(f)
        v = cls(k=data["k"], levels=data["levels"])
        v._centers  = data["centers"]
        v._idf      = data["idf"]
        v._doc_freq = data["doc_freq"]
        v._n_docs   = data["n_docs"]
        v.is_built  = True
        print(f"[Vocab] Loaded from {path}  ({v.vocab_size} words, {v._n_docs} docs)")
        return v

    def __repr__(self):
        return (f"VisualVocabulary(k={self.k}, L={self.levels}, "
                f"words={self.vocab_size}, built={self.is_built})")
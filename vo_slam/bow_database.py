"""
bow_database.py
---------------
BoW database with inverted index for fast place recognition queries.

Inverted index
--------------
  word_id → [(kf_id, weight), ...]

Query algorithm (L1 score, ORB-SLAM style)
------------------------------------------
For a query BoW vector q and database entry d:
  score(q, d) = 1 - 0.5 * Σ |q_i - d_i|

Complexity: O(W) per word in query, not O(|database|).
The inverted index means we only compare against KFs that share words.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
from .vocabulary import BowVector, VisualVocabulary


@dataclass
class QueryResult:
    kf_id    : int
    score    : float

    def __repr__(self):
        return f"QueryResult(kf_id={self.kf_id}, score={self.score:.4f})"


class BowDatabase:
    """
    Inverted-index BoW database.

    Usage
    -----
    db = BowDatabase(vocab)
    db.add(bow_vec)            # add a keyframe's BoW vector
    results = db.query(bow_vec, max_results=5)
    """

    def __init__(self, vocab: VisualVocabulary):
        self.vocab   = vocab
        # word_id → list of (kf_id, weight)
        self._index  : Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        self._bows   : Dict[int, BowVector] = {}     # kf_id → BowVector
        self._n_entries = 0

    # ------------------------------------------------------------------ #
    #  Insert                                                              #
    # ------------------------------------------------------------------ #

    def add(self, bow: BowVector) -> None:
        """Add a keyframe's BoW vector to the database."""
        self._bows[bow.kf_id] = bow
        self.vocab.add_document_words(bow)

        # Update inverted index
        for word_id, weight in zip(bow.word_ids, bow.weights):
            self._index[int(word_id)].append((bow.kf_id, float(weight)))

        self._n_entries += 1

        # Recompute IDF every 10 insertions
        if self._n_entries % 10 == 0:
            self.vocab.update_idf()

    # ------------------------------------------------------------------ #
    #  Query                                                              #
    # ------------------------------------------------------------------ #

    def query(
        self,
        bow         : BowVector,
        max_results : int = 5,
        exclude_ids : Optional[Set[int]] = None,
    ) -> List[QueryResult]:
        """
        Find the most similar keyframes to the query.

        Parameters
        ----------
        bow         : query BoW vector
        max_results : number of results to return
        exclude_ids : kf_ids to exclude (e.g. temporal neighbors)

        Returns
        -------
        List of QueryResult sorted by score descending.
        """
        if self._n_entries == 0:
            return []

        exclude = exclude_ids or set()

        # Accumulate scores via inverted index
        # Only compare against KFs that share at least one word
        score_accum: Dict[int, float] = defaultdict(float)

        q_dict = bow.to_dict()
        for word_id, q_weight in q_dict.items():
            for (kf_id, db_weight) in self._index.get(word_id, []):
                if kf_id in exclude:
                    continue
                # Partial L1 score accumulation
                score_accum[kf_id] += min(q_weight, db_weight)

        if not score_accum:
            return []

        # Finalise L1 scores
        # L1 score = 1 - 0.5 * |q - d|_1
        # Since both vectors are L1-normalised:
        # |q - d|_1 = 2 - 2 * Σ min(q_i, d_i)
        # So score = Σ min(q_i, d_i)  (already computed above)
        results = [
            QueryResult(kf_id=kf_id, score=float(score))
            for kf_id, score in score_accum.items()
        ]
        results.sort(key=lambda r: -r.score)
        return results[:max_results]

    # ------------------------------------------------------------------ #
    #  Direct lookup                                                       #
    # ------------------------------------------------------------------ #

    def get_bow(self, kf_id: int) -> Optional[BowVector]:
        return self._bows.get(kf_id)

    def score_pair(self, kf_id_a: int, kf_id_b: int) -> float:
        """Direct L1 score between two stored keyframes."""
        a = self._bows.get(kf_id_a)
        b = self._bows.get(kf_id_b)
        if a is None or b is None:
            return 0.0
        return a.l1_score(b)

    def __len__(self) -> int:
        return self._n_entries

    def __repr__(self):
        return f"BowDatabase({self._n_entries} entries)"
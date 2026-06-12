"""
place_recognition.py
--------------------
High-level place recognition system.

Algorithm (mirrors ORB-SLAM2 loop detection)
--------------------------------------------
1. Transform current KF descriptors → BoW vector
2. Query database for top candidates
3. Filter candidates:
   a. Exclude temporal neighbors (KFs adjacent in the co-visibility graph)
   b. Apply minimum score threshold
   c. Group by co-visible keyframes, take best group score
4. Consistency check:
   - A candidate must pass detection in 3 consecutive KF queries
   - This eliminates false positives from visually similar but different places
5. Return verified loop candidate(s)

Thresholds
----------
  min_score       : 0.012   (ORB-SLAM2 default for ORB)
  consistency     : 3       (consecutive detections required)
  temporal_window : 20      (exclude this many recent KFs from query)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from .vocabulary   import VisualVocabulary, BowVector
from .bow_database import BowDatabase, QueryResult


@dataclass
class LoopCandidate:
    """A verified loop closure candidate."""
    query_kf_id     : int       # current keyframe
    match_kf_id     : int       # detected loop keyframe
    score           : float     # BoW similarity score
    consecutive_hits: int       # how many consecutive KFs agreed


class PlaceRecognizer:
    """
    Detects loop closure candidates using BoW place recognition
    with temporal consistency filtering.

    Parameters
    ----------
    vocab            : built VisualVocabulary
    min_score        : minimum BoW similarity to consider a candidate
    consistency      : how many consecutive KF queries must agree
    temporal_window  : number of recent KFs to exclude from queries
    """

    def __init__(
        self,
        vocab           : VisualVocabulary,
        min_score       : float = 0.012,
        consistency     : int   = 3,
        temporal_window : int   = 20,
    ):
        self.vocab            = vocab
        self.min_score        = min_score
        self.consistency      = consistency
        self.temporal_window  = temporal_window

        self.db               = BowDatabase(vocab)

        # Consistency tracking: candidate_kf_id → consecutive hit count
        self._candidate_hits  : Dict[int, int]   = {}
        self._last_candidates : Set[int]         = set()

        # Statistics
        self.n_queries        = 0
        self.n_candidates     = 0
        self.n_loops_detected = 0

        # History of all KF ids (in insertion order) for temporal filtering
        self._kf_id_history   : List[int] = []

    # ------------------------------------------------------------------ #
    #  Add keyframe to database                                            #
    # ------------------------------------------------------------------ #

    def add_keyframe(
        self,
        kf_id       : int,
        descriptors : np.ndarray,   # (N, 32) uint8 ORB descriptors
    ) -> BowVector:
        """
        Convert descriptors to BoW and add to database.
        Returns the BoW vector (store it in the Keyframe for later use).
        """
        bow = self.vocab.transform(descriptors, kf_id=kf_id)
        self.db.add(bow)
        self._kf_id_history.append(kf_id)
        return bow

    # ------------------------------------------------------------------ #
    #  Query                                                               #
    # ------------------------------------------------------------------ #

    def detect_loop(
        self,
        kf_id       : int,
        descriptors : np.ndarray,
        covis_ids   : Optional[Set[int]] = None,   # co-visible neighbor KF ids
    ) -> Optional[LoopCandidate]:
        """
        Query the database for loop candidates.
        Returns a LoopCandidate if consistency check passes, else None.

        Parameters
        ----------
        kf_id       : current keyframe id
        descriptors : current KF's ORB descriptors
        covis_ids   : co-visibility neighbors (excluded from query)
        """
        if len(self.db) < self.temporal_window + 5:
            return None   # not enough history yet

        self.n_queries += 1

        # ── 1. BoW vector for current KF ─────────────────────────────── #
        bow = self.vocab.transform(descriptors, kf_id=kf_id)

        # ── 2. Build exclusion set ────────────────────────────────────── #
        # Exclude recent KFs (temporal window) + co-visible neighbors
        recent_ids = set(self._kf_id_history[-self.temporal_window:])
        exclude    = recent_ids | (covis_ids or set()) | {kf_id}

        # ── 3. Query database ─────────────────────────────────────────── #
        raw_results = self.db.query(bow, max_results=10, exclude_ids=exclude)
        if not raw_results:
            self._update_consistency(set())
            return None

        # ── 4. Score threshold filter ─────────────────────────────────── #
        # Baseline: use best score of recent KF against DB as reference
        baseline  = self._compute_baseline_score(bow, exclude)
        threshold = max(self.min_score, 0.75 * baseline)

        candidates = [r for r in raw_results if r.score >= threshold]
        if not candidates:
            self._update_consistency(set())
            return None

        # ── 5. Group by co-visible keyframes ──────────────────────────── #
        # Candidates that are co-visible to each other → pick best group
        best = candidates[0]   # sorted by score desc

        # ── 6. Temporal consistency check ─────────────────────────────── #
        current_candidate_ids = {r.kf_id for r in candidates}
        consistent_id         = self._update_consistency(current_candidate_ids)

        if consistent_id is None:
            self.n_candidates += 1
            return None   # not yet consistent

        # ── 7. Verified loop candidate ───────────────────────────────── #
        score = self.db.score_pair(kf_id, consistent_id) if kf_id in [r.kf_id for r in raw_results] \
                else best.score

        self.n_loops_detected += 1
        return LoopCandidate(
            query_kf_id      = kf_id,
            match_kf_id      = consistent_id,
            score            = best.score,
            consecutive_hits = self._candidate_hits.get(consistent_id, self.consistency),
        )

    # ------------------------------------------------------------------ #
    #  Baseline score                                                      #
    # ------------------------------------------------------------------ #

    def _compute_baseline_score(
        self,
        bow     : BowVector,
        exclude : Set[int],
    ) -> float:
        """
        Score of current KF against its recent temporal neighbors.
        Used to set a relative threshold — nearby KFs should score high,
        loop candidates must score at least 75% of this.
        """
        recent_included = [
            kf_id for kf_id in self._kf_id_history[-5:]
            if kf_id not in exclude
        ]
        if not recent_included:
            return self.min_score

        scores = []
        for kf_id in recent_included:
            stored = self.db.get_bow(kf_id)
            if stored:
                scores.append(bow.l1_score(stored))

        return float(np.mean(scores)) if scores else self.min_score

    # ------------------------------------------------------------------ #
    #  Consistency tracker                                                 #
    # ------------------------------------------------------------------ #

    def _update_consistency(
        self,
        current_candidates : Set[int],
    ) -> Optional[int]:
        """
        Update hit counters.
        Returns the candidate_id that reached consistency threshold, or None.
        """
        # Increment hits for candidates seen in both this and last query
        new_hits: Dict[int, int] = {}
        for cid in current_candidates:
            if cid in self._last_candidates:
                new_hits[cid] = self._candidate_hits.get(cid, 1) + 1
            else:
                new_hits[cid] = 1

        self._candidate_hits  = new_hits
        self._last_candidates = current_candidates

        # Check if any candidate reached the threshold
        verified = [
            (cid, hits) for cid, hits in new_hits.items()
            if hits >= self.consistency
        ]
        if not verified:
            return None

        # Return highest-scoring consistent candidate
        verified.sort(key=lambda x: -x[1])
        return verified[0][0]

    # ------------------------------------------------------------------ #
    #  Stats                                                               #
    # ------------------------------------------------------------------ #

    def summary(self) -> str:
        return (
            f"PlaceRecognizer | "
            f"DB={len(self.db)} entries | "
            f"queries={self.n_queries} | "
            f"candidates={self.n_candidates} | "
            f"loops={self.n_loops_detected}"
        )
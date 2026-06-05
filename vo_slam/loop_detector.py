"""
loop_detector.py
----------------
Loop Closing Thread — Stage 3 of the VSLAM pipeline.

What it does
------------
1. Receives new keyframes from VO via a queue
2. Builds BoW vector and adds to database
3. Queries for loop candidates
4. Geometric verification via feature matching + Essential matrix
5. Fires on_loop_detected(query_kf, match_kf, T_match_query) callback

Geometric verification
----------------------
BoW gives photometric similarity.
Geometric check confirms spatial consistency:
  - Match ORB descriptors between query KF and loop KF
  - Compute Essential matrix via RANSAC
  - Require min_geo_inliers (default 30) to accept

The callback on_loop_detected receives:
  query_kf  : current keyframe
  match_kf  : the detected loop keyframe
  T_rel     : 4×4 SE3 — relative pose from match→query (for correction)

This callback is where Sim3 optimisation and pose-graph correction
(Stage 4) will be wired in.
"""

from __future__ import annotations
import queue
import threading
import time
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set
from .vocabulary        import VisualVocabulary
from .place_recognition import PlaceRecognizer, LoopCandidate


# ═══════════════════════════════════════════════════════════════════════ #
#  Vocabulary builder helper                                              #
# ═══════════════════════════════════════════════════════════════════════ #

def build_vocabulary_from_keyframes(
    keyframes,                     # List[Keyframe]
    vocab_size    : int   = 1000,
    save_path     : Optional[str] = None,
    verbose       : bool  = True,
) -> VisualVocabulary:
    """
    Build a VisualVocabulary from existing keyframes' descriptors.
    Call this after the first N keyframes have been collected.
    """
    all_descs = []
    for kf in keyframes:
        if kf.features.descriptors is not None and len(kf.features.descriptors) > 0:
            all_descs.append(kf.features.descriptors)

    if not all_descs:
        raise ValueError("No descriptors found in keyframes")

    pool = np.vstack(all_descs)
    if verbose:
        print(f"[VocabBuilder] {len(pool)} descriptors from {len(keyframes)} KFs")

    # Determine k and levels for desired vocab_size
    # vocab_size = k^levels
    # Use k=10 → levels = ceil(log10(vocab_size))
    import math
    k      = 10
    levels = max(2, round(math.log(vocab_size, k)))
    actual = k ** levels

    vocab = VisualVocabulary(k=k, levels=levels)
    vocab.build(pool, verbose=verbose)

    if save_path:
        vocab.save(save_path)

    return vocab


# ═══════════════════════════════════════════════════════════════════════ #
#  Loop detection event                                                   #
# ═══════════════════════════════════════════════════════════════════════ #

@dataclass
class LoopEvent:
    """Emitted when a loop is geometrically verified."""
    query_kf_id  : int
    match_kf_id  : int
    bow_score    : float
    geo_inliers  : int
    T_rel        : np.ndarray    # 4×4 SE3 — match_cam → query_cam


# ═══════════════════════════════════════════════════════════════════════ #
#  Loop Detector                                                          #
# ═══════════════════════════════════════════════════════════════════════ #

class LoopDetector:
    """
    Background thread for place recognition + geometric verification.

    Parameters
    ----------
    vocab              : VisualVocabulary (must be built before starting)
    camera_K           : 3×3 intrinsic matrix
    on_loop_detected   : callback(LoopEvent) — fires on verified loop
    min_bow_score      : minimum BoW score to proceed to geo verification
    min_geo_inliers    : minimum RANSAC inliers to confirm loop
    consistency        : consecutive BoW detections required
    temporal_window    : recent KFs excluded from query
    vocab_build_at     : build vocabulary after this many KFs (if not pre-built)
    verbose            : print detection events
    """

    def __init__(
        self,
        vocab             : Optional[VisualVocabulary],
        camera_K          : np.ndarray,
        on_loop_detected  : Optional[Callable[[LoopEvent], None]] = None,
        min_bow_score     : float = 0.012,
        min_geo_inliers   : int   = 30,
        consistency       : int   = 3,
        temporal_window   : int   = 20,
        vocab_build_at    : int   = 50,     # build vocab after this many KFs
        verbose           : bool  = True,
    ):
        self.vocab            = vocab
        self.camera_K         = camera_K
        self.on_loop_detected = on_loop_detected
        self.min_bow_score    = min_bow_score
        self.min_geo_inliers  = min_geo_inliers
        self.verbose          = verbose
        self.vocab_build_at   = vocab_build_at

        self._recognizer  : Optional[PlaceRecognizer] = None
        if vocab is not None and vocab.is_built:
            self._recognizer = PlaceRecognizer(
                vocab,
                min_score       = min_bow_score,
                consistency     = consistency,
                temporal_window = temporal_window,
            )

        # KF buffer for vocabulary building
        self._kf_buffer   : list  = []
        self._kf_map      : dict  = {}    # kf_id → Keyframe

        self._queue       = queue.Queue()
        self._thread      : Optional[threading.Thread] = None
        self._stop_flag   = threading.Event()

        # Stats
        self.loop_events  : List[LoopEvent] = []
        self.n_bow_hits   = 0
        self.n_geo_verified = 0

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def start(self):
        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="LoopDetector")
        self._thread.start()
        print("[LoopDetector] started")

    def stop(self):
        self._stop_flag.set()
        self._queue.put(None)
        if self._thread:
            self._thread.join(timeout=5.0)
        if self._recognizer:
            print(f"[LoopDetector] stopped | {self._recognizer.summary()}")

    def enqueue(self, kf) -> None:
        """Called by VO on_new_keyframe hook. Non-blocking."""
        self._queue.put(kf)

    # ------------------------------------------------------------------ #
    #  Background thread                                                   #
    # ------------------------------------------------------------------ #

    def _run(self):
        while not self._stop_flag.is_set():
            try:
                kf = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if kf is None:
                break
            self._process_keyframe(kf)

    def _process_keyframe(self, kf):
        """Main processing for one keyframe."""
        self._kf_buffer.append(kf)
        self._kf_map[kf.kf_id] = kf

        descs = kf.features.descriptors
        if descs is None or len(descs) == 0:
            return

        # ── Build vocabulary if not ready ────────────────────────────── #
        if self._recognizer is None:
            if len(self._kf_buffer) >= self.vocab_build_at:
                self._build_vocab_and_init()
            return  # wait until vocab is built

        # ── Add current KF to database ───────────────────────────────── #
        bow = self._recognizer.add_keyframe(kf.kf_id, descs)

        # ── Get co-visible neighbor ids ──────────────────────────────── #
        covis_ids: Set[int] = set()
        if hasattr(kf, 'map_points') and kf.map_points:
            # Simple proxy: all KFs that share map points (if available)
            pass  # populated from covis_graph if wired in

        # ── Query for loop candidate ─────────────────────────────────── #
        candidate = self._recognizer.detect_loop(
            kf_id       = kf.kf_id,
            descriptors = descs,
            covis_ids   = covis_ids,
        )

        if candidate is None:
            return

        self.n_bow_hits += 1
        if self.verbose:
            print(f"  [LoopDetector] BoW candidate: "
                  f"KF{candidate.query_kf_id} ↔ KF{candidate.match_kf_id}  "
                  f"score={candidate.score:.4f}  hits={candidate.consecutive_hits}")

        # ── Geometric verification ───────────────────────────────────── #
        match_kf = self._kf_map.get(candidate.match_kf_id)
        if match_kf is None:
            return

        event = self._geometric_verify(kf, match_kf, candidate)
        if event is None:
            return

        self.n_geo_verified += 1
        self.loop_events.append(event)

        print(f"  [LoopDetector] ✓ LOOP VERIFIED: "
              f"KF{event.query_kf_id} ↔ KF{event.match_kf_id}  "
              f"inliers={event.geo_inliers}  score={event.bow_score:.4f}")

        if self.on_loop_detected:
            self.on_loop_detected(event)

    # ------------------------------------------------------------------ #
    #  Geometric verification                                              #
    # ------------------------------------------------------------------ #

    def _geometric_verify(
        self,
        query_kf,
        match_kf,
        candidate : LoopCandidate,
    ) -> Optional[LoopEvent]:
        """
        Verify loop geometrically via:
          1. ORB feature matching between query and match KF
          2. Essential matrix estimation via RANSAC
          3. Require min_geo_inliers

        Returns LoopEvent if verified, None otherwise.
        """
        q_descs = query_kf.features.descriptors
        m_descs = match_kf.features.descriptors
        q_pts   = query_kf.features.pts2d
        m_pts   = match_kf.features.pts2d

        if q_descs is None or m_descs is None:
            return None
        if len(q_descs) < 20 or len(m_descs) < 20:
            return None

        # BF matching with ratio test
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        try:
            raw = matcher.knnMatch(q_descs, m_descs, k=2)
        except cv2.error:
            return None

        good = []
        for pair in raw:
            if len(pair) == 2:
                m, n = pair
                if m.distance < 0.75 * n.distance:
                    good.append(m)

        if len(good) < self.min_geo_inliers:
            return None

        pts_q = np.array([q_pts[m.queryIdx] for m in good], dtype=np.float32)
        pts_m = np.array([m_pts[m.trainIdx] for m in good], dtype=np.float32)

        # Essential matrix
        E, mask = cv2.findEssentialMat(
            pts_q, pts_m, self.camera_K,
            method    = cv2.RANSAC,
            prob      = 0.999,
            threshold = 1.0,
        )
        if E is None or mask is None:
            return None

        n_inliers = int(mask.sum())
        if n_inliers < self.min_geo_inliers:
            return None

        # Recover relative pose
        _, R, t, pose_mask = cv2.recoverPose(
            E, pts_q, pts_m, self.camera_K,
            mask=mask.copy()
        )

        # Build T_rel (match → query, in match camera frame)
        T_rel = np.eye(4)
        T_rel[:3, :3] = R
        T_rel[:3,  3] = t.ravel()

        return LoopEvent(
            query_kf_id = candidate.query_kf_id,
            match_kf_id = candidate.match_kf_id,
            bow_score   = candidate.score,
            geo_inliers = n_inliers,
            T_rel       = T_rel,
        )

    # ------------------------------------------------------------------ #
    #  Vocabulary builder                                                  #
    # ------------------------------------------------------------------ #

    def _build_vocab_and_init(self):
        print(f"\n[LoopDetector] Building vocabulary from "
              f"{len(self._kf_buffer)} keyframes...")
        t0 = time.perf_counter()

        try:
            vocab = build_vocabulary_from_keyframes(
                self._kf_buffer,
                vocab_size = 1000,
                verbose    = True,
            )
        except Exception as e:
            print(f"[LoopDetector] Vocabulary build failed: {e}")
            return

        self._recognizer = PlaceRecognizer(
            vocab,
            min_score       = self.min_bow_score,
            temporal_window = 20,
        )

        # Add all buffered KFs to the database
        for kf in self._kf_buffer:
            if kf.features.descriptors is not None:
                self._recognizer.add_keyframe(kf.kf_id, kf.features.descriptors)

        dt = time.perf_counter() - t0
        print(f"[LoopDetector] Vocabulary ready in {dt:.1f}s  "
              f"({vocab.vocab_size} words)")

    # ------------------------------------------------------------------ #
    #  Stats                                                               #
    # ------------------------------------------------------------------ #

    def summary(self) -> str:
        return (
            f"LoopDetector | "
            f"BoW hits={self.n_bow_hits} | "
            f"Verified={self.n_geo_verified} | "
            f"Events={len(self.loop_events)}"
        )
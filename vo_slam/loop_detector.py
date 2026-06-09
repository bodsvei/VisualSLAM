"""
loop_detector.py  (geometric verification fix)
-----------------------------------------------
Why Verified=0 with 526 BoW hits
----------------------------------
1. min_geo_inliers=30 is too strict for frames with moderate overlap.
   Lowered to 15 — still enough to confirm a genuine loop.

2. Ratio test threshold 0.75 too tight across large viewpoint changes.
   Raised to 0.80 for loop candidate matching only.

3. Callback exception handling preserved from previous fix.

4. Dead zone and match-KF deduplication preserved.
"""

from __future__ import annotations
import queue
import threading
import time
import traceback
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set
from .vocabulary        import VisualVocabulary
from .place_recognition import PlaceRecognizer, LoopCandidate


def build_vocabulary_from_keyframes(
    keyframes,
    vocab_size  : int  = 1000,
    save_path   : Optional[str] = None,
    verbose     : bool = True,
) -> VisualVocabulary:
    all_descs = []
    for kf in keyframes:
        if kf.features.descriptors is not None and len(kf.features.descriptors) > 0:
            all_descs.append(kf.features.descriptors)
    if not all_descs:
        raise ValueError("No descriptors found in keyframes")
    pool = np.vstack(all_descs)
    if verbose:
        print(f"[VocabBuilder] {len(pool)} descriptors from {len(keyframes)} KFs")
    import math
    k      = 10
    levels = max(2, round(math.log(vocab_size, k)))
    vocab  = VisualVocabulary(k=k, levels=levels)
    vocab.build(pool, verbose=verbose)
    if save_path:
        vocab.save(save_path)
    return vocab


@dataclass
class LoopEvent:
    query_kf_id : int
    match_kf_id : int
    bow_score   : float
    geo_inliers : int
    T_rel       : np.ndarray


class LoopDetector:
    """
    Parameters
    ----------
    min_geo_inliers     : RANSAC inlier threshold for geometric verification
                          (lowered from 30 → 15)
    geo_ratio_thresh    : Lowe ratio for loop candidate matching
                          (raised from 0.75 → 0.80 for viewpoint tolerance)
    min_loop_gap_frames : dead zone — suppress callbacks for N frames after
                          a verified loop
    """

    def __init__(
        self,
        vocab               : Optional[VisualVocabulary],
        camera_K            : np.ndarray,
        on_loop_detected    : Optional[Callable[[LoopEvent], None]] = None,
        min_bow_score       : float = 0.012,
        min_geo_inliers     : int   = 15,       # ← lowered from 30
        geo_ratio_thresh    : float = 0.80,     # ← raised from 0.75
        consistency         : int   = 3,
        temporal_window     : int   = 20,
        vocab_build_at      : int   = 50,
        min_loop_gap_frames : int   = 200,
        verbose             : bool  = True,
    ):
        self.vocab               = vocab
        self.camera_K            = camera_K
        self.on_loop_detected    = on_loop_detected
        self.min_bow_score       = min_bow_score
        self.min_geo_inliers     = min_geo_inliers
        self.geo_ratio_thresh    = geo_ratio_thresh
        self.verbose             = verbose
        self.vocab_build_at      = vocab_build_at
        self.min_loop_gap_frames = min_loop_gap_frames

        self._recognizer : Optional[PlaceRecognizer] = None
        if vocab is not None and vocab.is_built:
            self._recognizer = PlaceRecognizer(
                vocab,
                min_score       = min_bow_score,
                consistency     = consistency,
                temporal_window = temporal_window,
            )

        self._kf_buffer : list = []
        self._kf_map    : dict = {}

        # Dead zone + dedup
        self._last_loop_frame   : int      = -9999
        self._used_match_kf_ids : Set[int] = set()

        self._queue     = queue.Queue()
        self._thread    : Optional[threading.Thread] = None
        self._stop_flag = threading.Event()

        self.loop_events    : List[LoopEvent] = []
        self.n_bow_hits     = 0
        self.n_geo_verified = 0
        self.n_suppressed   = 0
        self.n_geo_failed   = 0   # track geo failures for diagnosis

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def start(self):
        self._stop_flag.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="LoopDetector"
        )
        self._thread.start()
        print(f"[LoopDetector] started  "
              f"min_geo_inliers={self.min_geo_inliers}  "
              f"ratio={self.geo_ratio_thresh}")

    def stop(self):
        self._stop_flag.set()
        self._queue.put(None)
        if self._thread:
            self._thread.join(timeout=5.0)
        print(f"[LoopDetector] stopped | {self.summary()}")

    def enqueue(self, kf) -> None:
        self._queue.put(kf)

    # ------------------------------------------------------------------ #
    #  Thread loop                                                         #
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
        self._kf_buffer.append(kf)
        self._kf_map[kf.kf_id] = kf

        descs = kf.features.descriptors
        if descs is None or len(descs) == 0:
            return

        # Build vocabulary when ready
        if self._recognizer is None:
            if len(self._kf_buffer) >= self.vocab_build_at:
                self._build_vocab_and_init()
            return

        # Add to DB
        self._recognizer.add_keyframe(kf.kf_id, descs)

        # Query
        candidate = self._recognizer.detect_loop(
            kf_id       = kf.kf_id,
            descriptors = descs,
            covis_ids   = set(),
        )
        if candidate is None:
            return

        self.n_bow_hits += 1

        # Dead zone check
        if kf.frame_id - self._last_loop_frame < self.min_loop_gap_frames:
            self.n_suppressed += 1
            return

        # Match-KF dedup
        if candidate.match_kf_id in self._used_match_kf_ids:
            self.n_suppressed += 1
            return

        if self.verbose:
            print(f"  [LoopDetector] BoW candidate: "
                  f"KF{candidate.query_kf_id} ↔ KF{candidate.match_kf_id}  "
                  f"score={candidate.score:.4f}")

        # Geometric verification
        match_kf = self._kf_map.get(candidate.match_kf_id)
        if match_kf is None:
            return

        event = self._geometric_verify(kf, match_kf, candidate)
        if event is None:
            self.n_geo_failed += 1
            if self.verbose:
                print(f"  [LoopDetector] Geo verification FAILED for "
                      f"KF{candidate.query_kf_id} ↔ KF{candidate.match_kf_id}")
            return

        # Register
        self.n_geo_verified     += 1
        self._last_loop_frame    = kf.frame_id
        self._used_match_kf_ids.add(candidate.match_kf_id)
        self.loop_events.append(event)

        print(f"  [LoopDetector] ✓ LOOP VERIFIED: "
              f"KF{event.query_kf_id} ↔ KF{event.match_kf_id}  "
              f"inliers={event.geo_inliers}  score={event.bow_score:.4f}")

        if self.on_loop_detected:
            try:
                self.on_loop_detected(event)
            except Exception:
                print(f"[LoopDetector] Exception in on_loop_detected callback:")
                traceback.print_exc()
                print(f"[LoopDetector] Thread continuing.")

    # ------------------------------------------------------------------ #
    #  Geometric verification                                              #
    # ------------------------------------------------------------------ #

    def _geometric_verify(
        self,
        query_kf,
        match_kf,
        candidate,
    ) -> Optional[LoopEvent]:
        q_descs = query_kf.features.descriptors
        m_descs = match_kf.features.descriptors
        q_pts   = query_kf.features.pts2d
        m_pts   = match_kf.features.pts2d

        if q_descs is None or m_descs is None:
            return None
        if len(q_descs) < 15 or len(m_descs) < 15:
            return None

        # BF match with raised ratio threshold (more tolerant for loop KFs)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        try:
            raw = matcher.knnMatch(q_descs, m_descs, k=2)
        except cv2.error:
            return None

        good = []
        for pair in raw:
            if len(pair) == 2:
                m, n = pair
                if m.distance < self.geo_ratio_thresh * n.distance:
                    good.append(m)

        if len(good) < self.min_geo_inliers:
            return None

        pts_q = np.array([q_pts[m.queryIdx] for m in good], dtype=np.float32)
        pts_m = np.array([m_pts[m.trainIdx] for m in good], dtype=np.float32)

        E, mask = cv2.findEssentialMat(
            pts_q, pts_m, self.camera_K,
            method=cv2.RANSAC, prob=0.999, threshold=1.0,
        )
        if E is None or mask is None:
            return None

        n_inliers = int(mask.sum())
        if n_inliers < self.min_geo_inliers:
            return None

        _, R, t, _ = cv2.recoverPose(
            E, pts_q, pts_m, self.camera_K, mask=mask.copy()
        )

        R_cur_ref = R.T
        t_cur_ref = -(R.T @ t.ravel())
        T_rel     = np.eye(4)
        T_rel[:3, :3] = R_cur_ref
        T_rel[:3,  3] = t_cur_ref

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
                self._kf_buffer, vocab_size=1000, verbose=True,
            )
        except Exception as e:
            print(f"[LoopDetector] Vocabulary build failed: {e}")
            return

        self._recognizer = PlaceRecognizer(
            vocab, min_score=self.min_bow_score, temporal_window=20,
        )
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
            f"Geo failed={self.n_geo_failed} | "
            f"Verified={self.n_geo_verified} | "
            f"Suppressed={self.n_suppressed} | "
            f"Events fired={len(self.loop_events)}"
        )
"""
stereo_calibration.py
---------------------
Calibrates a stereo camera pair using synchronized chessboard images.
Finds:
  - Intrinsic matrices (K1, K2)
  - Distortion coefficients (D1, D2)
  - Extrinsic transformation (R, T) between cameras
  - Physical baseline (norm of T)
  - Rectification maps (for unwarping images into horizontal scanlines)

Usage:
    python3 stereo_calibration.py --left ./calib_images/left --right ./calib_images/right --rows 9 --cols 13 --square 20.0

Requirements:
    Images in --left and --right folders MUST be named identically or sorted
    such that images[i] from both folders form a synchronized pair.
"""

import argparse
import glob
import os
import sys
import cv2
import numpy as np
from typing import List, Optional, Tuple


def build_object_points(rows: int, cols: int, square_size: float) -> np.ndarray:
    objp = np.zeros((rows * cols, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size
    return objp


def calibrate_stereo(args):
    pattern_size = (args.cols, args.rows)
    objp_single = build_object_points(args.rows, args.cols, args.square)

    # 1. Gather image pairs
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    files_l = []
    for ext in exts:
        files_l.extend(glob.glob(os.path.join(args.left, ext)))
    files_l = sorted(files_l)

    if not files_l:
        sys.exit(f"[ERROR] No images found in {args.left}")

    obj_points = []  # 3D points in real world space
    img_points_l = [] # 2D points in image plane
    img_points_r = []
    img_shape = None

    print(f"Processing {len(files_l)} potential stereo pairs...")

    for path_l in files_l:
        name = os.path.basename(path_l)
        path_r = os.path.join(args.right, name)
        
        if not os.path.exists(path_r):
            print(f"  [SKIP] No matching right image for {name}")
            continue

        img_l = cv2.imread(path_l)
        img_r = cv2.imread(path_r)
        
        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        img_shape = gray_l.shape[::-1]

        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        ret_l, corners_l = cv2.findChessboardCorners(gray_l, pattern_size, flags)
        ret_r, corners_r = cv2.findChessboardCorners(gray_r, pattern_size, flags)

        if ret_l and ret_r:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
            corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)

            obj_points.append(objp_single)
            img_points_l.append(corners_l)
            img_points_r.append(corners_r)
            print(f"  [OK]   {name}")
        else:
            print(f"  [FAIL] Corners not found in both for {name}")

    if len(obj_points) < 5:
        sys.exit("[ERROR] Need at least 5 valid stereo pairs for reliable calibration.")

    # 2. Individual Calibration (Initial Guesses)
    print("\nCalibrating individual cameras...")
    ret_l, K1, D1, _, _ = cv2.calibrateCamera(obj_points, img_points_l, img_shape, None, None)
    ret_r, K2, D2, _, _ = cv2.calibrateCamera(obj_points, img_points_r, img_shape, None, None)
    print(f"  Left RMS:  {ret_l:.4f} px")
    print(f"  Right RMS: {ret_r:.4f} px")

    # 3. Stereo Calibration
    print("\nRunning stereo calibration (finding extrinsics)...")
    flags = cv2.CALIB_FIX_INTRINSIC
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

    ret_s, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        obj_points, img_points_l, img_points_r,
        K1, D1, K2, D2, img_shape,
        criteria=criteria_stereo, flags=flags
    )

    baseline = np.linalg.norm(T)
    print(f"\n  Stereo RMS Error: {ret_s:.6f} px")
    print(f"  Physical Baseline: {baseline:.4f} mm (based on square size)")
    
    if args.square == 1.0:
        print("  [NOTE] Square size was 1.0; baseline is in arbitrary units.")
    else:
        print(f"  Physical Baseline: {baseline/1000.0:.6f} meters")

    # 4. Rectification
    print("\nComputing rectification maps...")
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, img_shape, R, T)

    # 5. Save
    np.savez(args.save, 
             K1=K1, D1=D1, K2=K2, D2=D2, 
             R=R, T=T, R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
             rms=ret_s, baseline=baseline)
    print(f"Results saved to {args.save}")

    # Output for VisualOdometry config
    print("\n" + "="*40)
    print("  VisualOdometry Configuration")
    print("="*40)
    print(f"  fx: {K1[0,0]:.4f}")
    print(f"  fy: {K1[1,1]:.4f}")
    print(f"  cx: {K1[0,2]:.4f}")
    print(f"  cy: {K1[1,2]:.2f}")
    print(f"  baseline: {baseline/1000.0:.6f} (meters)")
    print("="*40 + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--left",   required=True, help="Folder with left images")
    p.add_argument("--right",  required=True, help="Folder with right images")
    p.add_argument("--rows",   type=int, default=6)
    p.add_argument("--cols",   type=int, default=9)
    p.add_argument("--square", type=float, default=25.0, help="Square size in mm")
    p.add_argument("--save",   default="stereo_calibration.npz")
    calibrate_stereo(p.parse_args())

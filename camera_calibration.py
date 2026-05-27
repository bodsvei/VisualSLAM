"""
camera_calibration.py
---------------------
Calibrates a camera using a chessboard pattern to find:
  - Intrinsic matrix K (camera matrix)
  - Distortion coefficients
  - Per-image reprojection error

Usage:
    python3 camera_calibration.py --images ./calib_images --rows 8 --cols 12 --square 20.0 --show --undistort

    --images   : folder containing calibration images (JPG/PNG)
    --rows     : number of INNER corners along rows (default: 6)
    --cols     : number of INNER corners along cols (default: 9)
    --square   : physical size of one square in mm (default: 25.0)
    --save     : path to save calibration results as .npz (default: calibration.npz)
    --show     : show detected corners in each image (flag)
    --undistort: show an undistorted version of the first image (flag)
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def build_object_points(rows: int, cols: int, square_size: float) -> np.ndarray:
    """
    Create the 3-D world coordinates of the chessboard corners.
    The board lies in the Z=0 plane.

    Returns shape (rows*cols, 1, 3) float32.
    """
    objp = np.zeros((rows * cols, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size
    return objp.reshape(-1, 1, 3)


def find_corners(image_path: str, pattern_size: tuple, show: bool = False):
    """
    Load an image, convert to grey, find chessboard corners.

    Returns:
        (corners_refined, image_shape)  if found
        None                             if not found
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"  [WARN] Cannot read image: {image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK
    )
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)

    if not found:
        print(f"  [SKIP] Corners not found in {os.path.basename(image_path)}")
        return None

    # Sub-pixel refinement
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    if show:
        vis = img.copy()
        cv2.drawChessboardCorners(vis, pattern_size, corners_refined, found)
        cv2.imshow(f"Corners – {os.path.basename(image_path)}", vis)
        cv2.waitKey(500)

    return corners_refined, gray.shape[::-1]  # shape as (w, h)


def compute_reprojection_error(
    obj_points_all, img_points_all, rvecs, tvecs, K, dist
) -> tuple:
    """
    Compute per-image and overall mean reprojection error.

    Returns:
        per_errors : list of per-image RMS errors
        mean_error : overall RMS across all points
    """
    per_errors = []
    total_sq = 0.0
    total_n = 0

    for objp, imgp, rvec, tvec in zip(obj_points_all, img_points_all, rvecs, tvecs):
        projected, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
        diff = imgp - projected
        sq_err = np.sum(diff ** 2, axis=-1)      # shape (N, 1)
        rms = float(np.sqrt(np.mean(sq_err)))
        per_errors.append(rms)
        total_sq += float(np.sum(sq_err))
        total_n  += sq_err.size

    mean_error = float(np.sqrt(total_sq / total_n))
    return per_errors, mean_error


def print_section(title: str) -> None:
    width = 60
    print("\n" + "─" * width)
    print(f"  {title}")
    print("─" * width)


# ──────────────────────────────────────────────────────────────────────────────
# Main calibration routine
# ──────────────────────────────────────────────────────────────────────────────

def calibrate(args) -> None:
    pattern_size = (args.cols, args.rows)          # (cols, rows) for OpenCV
    objp_template = build_object_points(args.rows, args.cols, args.square)

    # ── Gather images ────────────────────────────────────────────────────────
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif")
    image_files = []
    for ext in exts:
        image_files.extend(glob.glob(os.path.join(args.images, ext)))
        image_files.extend(glob.glob(os.path.join(args.images, ext.upper())))
    image_files = sorted(set(image_files))

    if not image_files:
        sys.exit(f"[ERROR] No images found in '{args.images}'. Check the path.")

    print_section(f"Found {len(image_files)} image(s)")

    # ── Detect corners ───────────────────────────────────────────────────────
    obj_points_all = []
    img_points_all = []
    img_shape      = None

    for path in image_files:
        result = find_corners(path, pattern_size, show=args.show)
        if result is None:
            continue
        corners, shape = result
        obj_points_all.append(objp_template)
        img_points_all.append(corners)
        img_shape = shape
        print(f"  [OK]   {os.path.basename(path)}")

    if args.show:
        cv2.destroyAllWindows()

    n_good = len(obj_points_all)
    print(f"\n  {n_good}/{len(image_files)} images used for calibration.")

    if n_good < 3:
        sys.exit("[ERROR] Need at least 3 valid images. Check pattern size / images.")

    # ── Run calibration ──────────────────────────────────────────────────────
    print_section("Running calibration …")

    rms_calib, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points_all,
        img_points_all,
        img_shape,
        None,
        None,
    )

    # ── Intrinsic matrix K ───────────────────────────────────────────────────
    print_section("Intrinsic Matrix  K")
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    skew   = K[0, 1]

    print(f"\n  K =\n{np.array2string(K, precision=4, suppress_small=True, prefix='      ')}\n")
    print(f"  Focal lengths  : fx = {fx:.4f} px,  fy = {fy:.4f} px")
    print(f"  Principal point: cx = {cx:.4f} px,  cy = {cy:.4f} px")
    print(f"  Skew           : {skew:.6f}")
    print(f"  Aspect ratio   : fy/fx = {fy/fx:.6f}")

    # ── Distortion coefficients ──────────────────────────────────────────────
    print_section("Distortion Coefficients  [k1, k2, p1, p2, k3]")
    labels = ["k1 (radial)", "k2 (radial)", "p1 (tangential)",
              "p2 (tangential)", "k3 (radial)"]
    d = dist.flatten()
    for label, val in zip(labels, d):
        print(f"  {label:<22}: {val:+.8f}")

    # ── Reprojection error ───────────────────────────────────────────────────
    print_section("Reprojection Error")
    per_errors, mean_error = compute_reprojection_error(
        obj_points_all, img_points_all, rvecs, tvecs, K, dist
    )

    print(f"\n  {'Image':<35} {'RMS error (px)'}")
    print(f"  {'─'*35} {'─'*15}")
    for path, err in zip(image_files, per_errors):
        flag = "  ← high" if err > 1.0 else ""
        print(f"  {os.path.basename(path):<35} {err:.6f}{flag}")

    print(f"\n  {'OpenCV calibrateCamera RMS':.<40} {rms_calib:.6f} px")
    print(f"  {'Manual mean RMS (all points)':.<40} {mean_error:.6f} px")

    quality = (
        "Excellent  (< 0.5 px)"   if mean_error < 0.5  else
        "Good       (0.5–1.0 px)" if mean_error < 1.0  else
        "Acceptable (1.0–2.0 px)" if mean_error < 2.0  else
        "Poor       (> 2.0 px) — retake images"
    )
    print(f"\n  Calibration quality: {quality}")

    # ── Field of view ─────────────────────────────────────────────────────────
    print_section("Field of View (diagonal / horizontal / vertical)")
    w, h    = img_shape
    fov_h   = 2 * np.degrees(np.arctan2(w / 2, fx))
    fov_v   = 2 * np.degrees(np.arctan2(h / 2, fy))
    fov_d   = 2 * np.degrees(np.arctan2(np.hypot(w / 2, h / 2), np.hypot(fx, fy)))
    print(f"\n  Horizontal FOV : {fov_h:.2f}°")
    print(f"  Vertical FOV   : {fov_v:.2f}°")
    print(f"  Diagonal FOV   : {fov_d:.2f}°")

    # ── Save results ──────────────────────────────────────────────────────────
    np.savez(
        args.save,
        K=K,
        dist=dist,
        rvecs=np.array(rvecs),
        tvecs=np.array(tvecs),
        rms=rms_calib,
        mean_reprojection_error=mean_error,
    )
    print_section(f"Results saved → {args.save}")

    # ── Optional undistortion preview ─────────────────────────────────────────
    if args.undistort and image_files:
        first_img = cv2.imread(image_files[0])
        if first_img is not None:
            h_img, w_img = first_img.shape[:2]
            new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w_img, h_img), 1)
            undistorted = cv2.undistort(first_img, K, dist, None, new_K)
            side_by_side = np.hstack([first_img, undistorted])
            cv2.imshow("Original (left)  vs  Undistorted (right)", side_by_side)
            print("\n  Press any key to close the preview window …")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    print()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Camera calibration — finds K matrix and reprojection error."
    )
    parser.add_argument(
        "--images", default="./calib_images",
        help="Directory containing calibration images (default: ./calib_images)"
    )
    parser.add_argument(
        "--rows", type=int, default=6,
        help="Number of inner corners along rows (default: 6)"
    )
    parser.add_argument(
        "--cols", type=int, default=9,
        help="Number of inner corners along cols (default: 9)"
    )
    parser.add_argument(
        "--square", type=float, default=25.0,
        help="Physical size of one square in mm (default: 25.0)"
    )
    parser.add_argument(
        "--save", default="calibration.npz",
        help="Output file path for calibration data (default: calibration.npz)"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display detected corners on each image (requires a display)"
    )
    parser.add_argument(
        "--undistort", action="store_true",
        help="Show undistorted version of the first image after calibration"
    )
    return parser.parse_args()


if __name__ == "__main__":
    calibrate(parse_args())
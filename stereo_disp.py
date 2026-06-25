import cv2
import numpy as np
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Compute and display stereo disparity using SGBM.")
    parser.add_argument("--kitti", type=str, default="KITTI", help="Path to KITTI dataset root directory")
    parser.add_argument("--seq", type=str, default="00", help="Sequence ID")
    parser.add_argument("--frame", type=int, default=0, help="Start frame index to process")
    parser.add_argument("--left", type=str, default="", help="Direct path to left image (overrides KITTI args)")
    parser.add_argument("--right", type=str, default="", help="Direct path to right image (overrides KITTI args)")
    parser.add_argument("--wait", type=int, default=1, help="Wait time between frames in ms. Set to 0 to pause on each frame.")
    
    args = parser.parse_args()
    
    l_imgs = []
    r_imgs = []
    
    if args.left and args.right:
        l_imgs = [Path(args.left)]
        r_imgs = [Path(args.right)]
    else:
        kitti_root = Path(args.kitti)
        img_l_dir = kitti_root / "sequences" / args.seq / "image_0"
        img_r_dir = kitti_root / "sequences" / args.seq / "image_1"
        
        # If KITTI doesn't exist, fallback
        if not kitti_root.exists():
            print(f"Warning: KITTI directory {kitti_root} not found. Trying calib_images...")
            img_l_dir = Path("calib_images/left")
            img_r_dir = Path("calib_images/right")
            
            if img_l_dir.exists() and img_r_dir.exists():
                l_imgs = sorted(img_l_dir.glob("*.png") + img_l_dir.glob("*.jpg"))
                r_imgs = sorted(img_r_dir.glob("*.png") + img_r_dir.glob("*.jpg"))
            else:
                print("Could not find calibration images either.")
                return
        else:
            l_imgs = sorted(img_l_dir.glob("*.png"))
            r_imgs = sorted(img_r_dir.glob("*.png"))
            
    if not l_imgs or not r_imgs:
        print("No images found.")
        return
        
    # SGBM Parameters (tuned for KITTI stereo)
    window_size = 5
    min_disp = 0
    num_disp = 16 * 6  # KITTI max disparity is usually around 96
    
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=window_size,
        P1=8 * 1 * window_size**2,
        P2=32 * 1 * window_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    print("Playing stereo sequence. Press 'q' to quit.")
    
    start_idx = min(args.frame, len(l_imgs) - 1)
    
    for i in range(start_idx, len(l_imgs)):
        img_l_path = l_imgs[i]
        
        if i >= len(r_imgs):
            break
        img_r_path = r_imgs[i]
        
        # Read images in grayscale
        img_l = cv2.imread(str(img_l_path), cv2.IMREAD_GRAYSCALE)
        img_r = cv2.imread(str(img_r_path), cv2.IMREAD_GRAYSCALE)
        
        if img_l is None or img_r is None:
            continue

        # Compute disparity
        disparity = stereo.compute(img_l, img_r).astype(np.float32) / 16.0

        # Filter out invalid disparity values (typically negative or less than min_disp)
        mask = disparity > min_disp
        
        # Normalize for visualization (only valid pixels)
        disp_vis = np.zeros_like(disparity, dtype=np.uint8)
        cv2.normalize(disparity, disp_vis, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U, mask=mask.astype(np.uint8))
        
        # Apply colormap to disparity
        disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
        
        # Mask out invalid disparity in color map (make them black)
        disp_color[~mask] = 0
        
        # Display using cv2.imshow for interactive loop
        img_l_color = cv2.cvtColor(img_l, cv2.COLOR_GRAY2BGR)
        
        # Stack left image and disparity map vertically for better fit on most screens
        combined = np.vstack((img_l_color, disp_color))
        
        cv2.imshow("Stereo Disparity - Press 'q' to quit", combined)
        
        # Determine wait time
        wait_time = args.wait if not (args.left and args.right) else 0
        key = cv2.waitKey(wait_time) & 0xFF
        
        if key == ord('q') or key == 27: # 'q' or ESC
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

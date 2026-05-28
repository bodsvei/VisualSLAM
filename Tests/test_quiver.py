import cv2
import numpy as np
from vo_slam import CameraModel, VisualOdometry, VOConfig

def main():
    camera = CameraModel.kitti() # just some camera
    cfg = VOConfig()
    vo = VisualOdometry(camera, cfg)
    
    cap = cv2.VideoCapture("Test-1.mov")
    for i in range(50):
        ret, frame = cap.read()
        if not ret: break
        stats = vo.process(frame)
        if stats.is_keyframe:
            d = vo.T_world_cam[:3, 2]
            print(f"Frame {i}: dir = {d}")

if __name__ == "__main__":
    main()

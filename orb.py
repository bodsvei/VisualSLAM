import argparse
import cv2
import numpy as np
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="ORB Feature Tracking")
    parser.add_argument("--webcam", action="store_true", help="Use webcam (device 0)")
    parser.add_argument("--video", type=str, default="Test-1.mov", help="Path to video file")
    args = parser.parse_args()

    if args.webcam:
        video_path = 0
    else:
        video_path = args.video
        if not os.path.exists(video_path):
            print(f"Warning: {video_path} not found in the current directory.")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        sys.exit(1)

    # Initialize ORB detector
    # nfeatures limits the number of best features to retain
    orb = cv2.ORB_create(nfeatures=1000)

    # Initialize Brute-Force Matcher with Hamming distance (standard for ORB)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to read the first frame.")
        sys.exit(1)

    # Convert to grayscale for detection
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Detect and compute for the first frame
    prev_kp, prev_des = orb.detectAndCompute(prev_gray, None)

    print("Starting tracking... Press 'q' to quit.")

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            print("End of video stream or failed to read frame.")
            break

        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect and compute for the current frame
        curr_kp, curr_des = orb.detectAndCompute(curr_gray, None)

        if prev_des is not None and curr_des is not None and len(prev_des) > 0 and len(curr_des) > 0:
            # Match descriptors using KNN (k=2 is required for Lowe's ratio test)
            matches = bf.knnMatch(prev_des, curr_des, k=2)

            # Apply Lowe's ratio test to filter out weak matches
            good_matches = []
            for m_n in matches:
                # Ensure we actually got 2 neighbors
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

            # Draw matches: Left is previous frame, Right is current frame
            # The function connects matched keypoints with lines
            out_img = cv2.drawMatches(
                prev_frame, prev_kp, 
                curr_frame, curr_kp, 
                good_matches, None, 
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                matchColor=(0, 255, 0) # Green lines
            )

            cv2.imshow("ORB Feature Tracking (Left: Prev, Right: Curr) | Press 'q' to quit", out_img)

        # Update previous frame and features for the next iteration
        prev_frame = curr_frame.copy()
        prev_gray = curr_gray
        prev_kp = curr_kp
        prev_des = curr_des

        # Wait 30ms before next frame; exit if 'q' is pressed
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

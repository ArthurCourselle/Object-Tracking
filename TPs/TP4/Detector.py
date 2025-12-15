"""
File name         : detectors.py
Description       : Object detector used for detecting the objects in a video /image
Python Version    : 3.7
"""

# Import python libraries
import numpy as np
import cv2


def detect(frame):
    # Convert frame from BGR to GRAY
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray', gray)

    # Edge detection using Canny function
    img_edges = cv2.Canny(gray, 50, 190, 3)
    # cv2.imshow('img_edges', img_edges)

    # Convert to black and white image
    ret, img_thresh = cv2.threshold(img_edges, 254, 255, cv2.THRESH_BINARY)
    # cv2.imshow('img_thresh', img_thresh)

    # Find contours
    contours, _ = cv2.findContours(
        img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Set the accepted minimum & maximum radius of a detected object
    min_radius_thresh = 3
    max_radius_thresh = 30

    centers = []
    for c in contours:
        # ref: https://docs.opencv.org/trunk/dd/d49/tutorial_py_contour_features.html
        (x, y), radius = cv2.minEnclosingCircle(c)
        radius = int(radius)

        # Take only the valid circle(s)
        if (radius > min_radius_thresh) and (radius < max_radius_thresh):
            centers.append(np.array([[x], [y]]))
    cv2.imshow("contours", img_thresh)
    return centers


def process_video(input_video_path, output_video_path):
    """
    Process a video file: detect centers in each frame and save a new video with centers displayed.

    Args:
        input_video_path (str): Path to the input video file
        output_video_path (str): Path to the output video file
    """

    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video file {input_video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"Cannot create output video file {output_video_path}")
        cap.release()
        return

    frame_count = 0
    print("Processing video...")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        centers = detect(frame)

        for center in centers:
            x = int(center[0, 0])
            y = int(center[1, 0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.line(frame, (x - 10, y), (x + 10, y), (0, 255, 0), 2)
            cv2.line(frame, (x, y - 10), (x, y + 10), (0, 255, 0), 2)

        out.write(frame)

        frame_count += 1
        if frame_count % 30 == 0 or frame_count == total_frames:
            print(f"Processed {frame_count}/{total_frames} frames")

    cap.release()
    out.release()
    print(f"Output saved to {output_video_path}")


if __name__ == "__main__":
    input_video = "video/randomball.avi"
    output_video = "video/output_with_centers.mp4"

    process_video(input_video, output_video)

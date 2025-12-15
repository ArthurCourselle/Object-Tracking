from Detector import detect
from KalmanFilter import KalmanFilter
import cv2
import numpy as np

kf = KalmanFilter(0.1, 1, 1, 1, 0.1, 0.1)


def process_video(input_video_path, output_video_path):

    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print(f"Cannot open video file {input_video_path}")
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
    trajectory = []
    initialized = False
    box_half = 20

    print("Processing video...")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        pred_state = kf.predict()
        pred_x = int(pred_state[0])
        pred_y = int(pred_state[1])

        centers = detect(frame)

        chosen_measurement = None
        if len(centers) > 0:
            if not initialized:
                mx = float(centers[0][0, 0])
                my = float(centers[0][1, 0])
                kf.x = np.array([mx, my, 0.0, 0.0])
                initialized = True
                chosen_measurement = np.array([mx, my])
            else:
                min_dist = None
                for c in centers:
                    cx = float(c[0, 0])
                    cy = float(c[1, 0])
                    dist = (cx - pred_x) ** 2 + (cy - pred_y) ** 2
                    if (min_dist is None) or (dist < min_dist):
                        min_dist = dist
                        chosen_measurement = np.array([cx, cy])

        estimated = None
        if chosen_measurement is not None:
            estimated_state = kf.update(chosen_measurement)
            est_x = int(estimated_state[0])
            est_y = int(estimated_state[1])
            estimated = (est_x, est_y)
            trajectory.append(estimated)
        else:
            estimated = (pred_x, pred_y)
            trajectory.append(estimated)

        # Draw all detected centers (green)
        for center in centers:
            x = int(center[0, 0])
            y = int(center[1, 0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.line(frame, (x - 10, y), (x + 10, y), (0, 255, 0), 2)
            cv2.line(frame, (x, y - 10), (x, y + 10), (0, 255, 0), 2)

        # Draw predicted rectangle (blue)
        top_left_pred = (pred_x - box_half, pred_y - box_half)
        bottom_right_pred = (pred_x + box_half, pred_y + box_half)
        cv2.rectangle(frame, top_left_pred, bottom_right_pred, (255, 0, 0), 2)

        # Draw estimated rectangle (red)
        est_x, est_y = estimated
        top_left_est = (est_x - box_half, est_y - box_half)
        bottom_right_est = (est_x + box_half, est_y + box_half)
        cv2.rectangle(frame, top_left_est, bottom_right_est, (0, 0, 255), 2)

        # Draw trajectory line (blue)
        if len(trajectory) > 1:
            pts = np.array(trajectory, dtype=np.int32)
            for i in range(1, len(pts)):
                cv2.line(frame, tuple(pts[i - 1]), tuple(pts[i]), (200, 100, 0), 2)

        out.write(frame)

        frame_count += 1
        if frame_count % 30 == 0 or frame_count == total_frames:
            print(f"Processed {frame_count}/{total_frames} frames")

    cap.release()
    out.release()
    print(f"Output saved to {output_video_path}")


if __name__ == "__main__":
    input_video = "video/randomball.avi"
    output_video = "video/output_with_kf.mp4"

    process_video(input_video, output_video)

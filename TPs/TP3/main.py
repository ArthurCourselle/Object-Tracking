import cv2
import numpy as np
import os
import sys

from kalman_iou_tracker import KalmanIOUTracker

def load_detections(det_file):

    detections = {}
    with open(det_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(',', ' ').split()
            if len(parts) < 7:
                 continue
                 
            frame = int(parts[0])
            x1 = float(parts[2])
            y1 = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            x2 = x1 + w
            y2 = y1 + h
            score = float(parts[6])
            
            if frame not in detections:
                detections[frame] = []
            detections[frame].append([x1, y1, x2, y2, score])
    
    return detections

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "../ADL-Rundle-6")
    
    img_dir = os.path.join(base_dir, "img1")
    det_file = os.path.join(base_dir, "det/Yolov5l/det.txt")
    output_video_path = os.path.join(script_dir, "tracking_output_tp3.mp4")
    output_txt_path = os.path.join(script_dir, "ADL-Rundle-6_tracking_tp3.txt")
    
    if not os.path.exists(det_file):
        print(f"Error: Detections file not found : {det_file}")
        return

    detections_map = load_detections(det_file)
    
    tracker = KalmanIOUTracker(max_age=30, min_hits=1, iou_threshold=0.3)
    
    out_video = None
    
    f_out = open(output_txt_path, 'w')
    
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    
    print(f"Processing {len(img_files)} frames with Kalman-IoU Tracker.")
    
    for i, img_file in enumerate(img_files):
        frame_id = i + 1
        img_path = os.path.join(img_dir, img_file)
        frame = cv2.imread(img_path)
        
        if frame is None:
            continue
            
        if out_video is None:
            height, width, _ = frame.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_video = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))
            
        dets = np.array(detections_map.get(frame_id, []))
        
        if len(dets) > 0:
             tracked_objects = tracker.update(dets)
        else:
             tracked_objects = tracker.update(np.empty((0, 5)))
             
        for trk in tracked_objects:
             x1, y1, x2, y2, trk_id = trk
             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
             trk_id = int(trk_id)
             
             color = (0, 255, 0)
             for t in tracker.tracks:
                 if t.id == trk_id:
                     color = t.color
                     break

             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
             cv2.putText(frame, str(trk_id), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
             
             w = x2 - x1
             h = y2 - y1
             f_out.write(f"{frame_id},{trk_id},{x1},{y1},{w},{h},1,-1,-1,-1\n")

        out_video.write(frame)
        
        if frame_id % 50 == 0:
            print(f"Processed frame {frame_id}/{len(img_files)}")
            
    f_out.close()
    if out_video:
        out_video.release()
    print("Tracking completed.")

if __name__ == "__main__":
    main()

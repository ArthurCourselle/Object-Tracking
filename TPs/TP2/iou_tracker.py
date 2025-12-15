import os
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

def iou_batch(bb_test, bb_gt):

    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
    return(o)

class IOUTracker:
    def __init__(self, max_age=15, min_hits=1, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = [] # List of (x1, y1, x2, y2, id, time_since_update, hit_streak)
        self.frame_count = 0
        self.id_count = 1

    def update(self, detections):
        
        self.frame_count += 1
        
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        
        for t, trk in enumerate(trks):
            pos = self.trackers[t][:4] # [x1, y1, x2, y2]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            
        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(detections, trks)
        
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                self.trackers[t][:4] = detections[d[0]][:4]
                self.trackers[t][5] = 0 # reset time_since_update
                self.trackers[t][6] += 1 # increment hit_streak
            else:
                 self.trackers[t][5] += 1 # increment time_since_update
        
        for i in unmatched_dets:
            trk = detections[i][:4]
            self.trackers.append([trk[0], trk[1], trk[2], trk[3], self.id_count, 0, 1]) 
            self.id_count += 1
            
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            i -= 1
            if trk[5] > self.max_age:
                self.trackers.pop(i)

        ret = []
        for trk in self.trackers:
            if (trk[5] < 1) and (trk[6] >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((trk[:4], [trk[4]])))
        return ret


    def associate_detections_to_trackers(self, detections, trackers):
        
        if (len(trackers)==0):
            return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

        iou_matrix = iou_batch(detections[:, :4], trackers[:, :4])

        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > self.iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                row_ind, col_ind = linear_sum_assignment(-iou_matrix) # maximize IoU
                matched_indices = np.stack((row_ind, col_ind), axis=1)
        else:
            matched_indices = np.empty((0,2))

        unmatched_detections = []
        for d, det in enumerate(detections):
            if(d not in matched_indices[:,0]):
                unmatched_detections.append(d)
        
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if(t not in matched_indices[:,1]):
                unmatched_trackers.append(t)

        matches = []
        for m in matched_indices:
            if(iou_matrix[m[0], m[1]] < self.iou_threshold):
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1,2))
        
        if(len(matches)==0):
            matches = np.empty((0,2),dtype=int)
        else:
            matches = np.concatenate(matches,axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


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
    output_video_path = os.path.join(script_dir, "tracking_output.mp4")
    output_txt_path = os.path.join(script_dir, "ADL-Rundle-6_tracking.txt")
    
    if not os.path.exists(det_file):
        print(f"Error: Detections file not found : {det_file}")
        return
        
    detections_map = load_detections(det_file)
    
    tracker = IOUTracker(max_age=15, min_hits=1, iou_threshold=0.3)
    
    out_video = None
    
    f_out = open(output_txt_path, 'w')
    
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    
    print(f"Processing {len(img_files)} frames.")
    
    for i, img_file in enumerate(img_files):
        frame_id = i + 1
        img_path = os.path.join(img_dir, img_file)
        frame = cv2.imread(img_path)
        
        if frame is None:
            print(f"Could not read image {img_path}")
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
             
             np.random.seed(trk_id)
             color = np.random.randint(0, 255, size=3).tolist()
             
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

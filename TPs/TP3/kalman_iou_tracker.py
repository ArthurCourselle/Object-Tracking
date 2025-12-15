import numpy as np
import sys
import os

from KalmanFilter import KalmanFilter
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

class Track:

    def __init__(self, detection, track_id):
        self.id = track_id
        self.bbox = detection
        self.hits = 1
        self.time_since_update = 0
        
        cx = (detection[0] + detection[2]) / 2
        cy = (detection[1] + detection[3]) / 2
        
        self.kf = KalmanFilter(dt=1, u_x=0, u_y=0, std_acc=0.1, x_std_meas=0.1, y_std_meas=0.1)
        self.kf.x = np.array([cx, cy, 0, 0])
        
        np.random.seed(self.id)
        self.color = np.random.randint(0, 255, size=3).tolist()

    def predict(self):
        self.kf.predict()
        
        cx = self.kf.x[0]
        cy = self.kf.x[1]
        
        w = self.bbox[2] - self.bbox[0]
        h = self.bbox[3] - self.bbox[1]
        
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
        
        return [x1, y1, x2, y2]
        
    def update(self, detection):
        self.time_since_update = 0
        self.hits += 1
        
        cx = (detection[0] + detection[2]) / 2
        cy = (detection[1] + detection[3]) / 2
        self.kf.update(np.array([cx, cy]))
        
        filtered_cx = self.kf.x[0]
        filtered_cy = self.kf.x[1]
        
        w = detection[2] - detection[0]
        h = detection[3] - detection[1]
        
        x1 = filtered_cx - w/2
        y1 = filtered_cy - h/2
        x2 = filtered_cx + w/2
        y2 = filtered_cy + h/2
        
        self.bbox = [x1, y1, x2, y2]


class KalmanIOUTracker:

    def __init__(self, max_age=15, min_hits=1, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.frame_count = 0
        self.id_count = 1

    def update(self, detections):

        self.frame_count += 1
        
        predicted_bboxes = []
        for t in self.tracks:
            predicted_bboxes.append(t.predict())
        
        predicted_bboxes = np.array(predicted_bboxes)
        
        if len(detections) > 0:
            det_bboxes = detections[:, :4]
        else:
            det_bboxes = np.empty((0, 4))
            
        if len(predicted_bboxes) > 0 and len(det_bboxes) > 0:
            iou_matrix = iou_batch(det_bboxes, predicted_bboxes)
        else:
            iou_matrix = np.empty((0,0))
            
        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > self.iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                row_ind, col_ind = linear_sum_assignment(-iou_matrix)
                matched_indices = np.stack((row_ind, col_ind), axis=1)
        else:
            matched_indices = np.empty((0,2))
            
        unmatched_detections = []
        for d, det in enumerate(det_bboxes):
            if(d not in matched_indices[:,0]):
                unmatched_detections.append(d)
        
        unmatched_tracks = []
        for t, trk in enumerate(self.tracks):
             if(t not in matched_indices[:,1]):
                unmatched_tracks.append(t)
                
        matches = []
        for m in matched_indices:
            if(iou_matrix[m[0], m[1]] < self.iou_threshold):
                unmatched_detections.append(m[0])
                unmatched_tracks.append(m[1])
            else:
                matches.append(m.reshape(1,2))
        
        if(len(matches)==0):
            matches = np.empty((0,2),dtype=int)
        else:
            matches = np.concatenate(matches,axis=0)

        for m in matches:
            det_idx = m[0]
            track_idx = m[1]
            self.tracks[track_idx].update(detections[det_idx][:4])
            
        for i in unmatched_detections:
            self.tracks.append(Track(detections[i][:4], self.id_count))
            self.id_count += 1
            
        for i in unmatched_tracks:
            self.tracks[i].time_since_update += 1
            
        i = len(self.tracks)
        for t in reversed(range(len(self.tracks))):
            if self.tracks[t].time_since_update > self.max_age:
                self.tracks.pop(t)
                
        ret = []
        for trk in self.tracks:
             if (trk.time_since_update < 1) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits):
                 ret.append(np.concatenate((trk.bbox, [trk.id]))) 
        return ret

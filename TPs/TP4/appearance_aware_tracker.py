import numpy as np
import cv2
import sys
import os
from scipy.optimize import linear_sum_assignment
from KalmanFilter import KalmanFilter

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

def cosine_similarity(a, b):
    norm_a = np.linalg.norm(a, axis=1, keepdims=True)
    norm_b = np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(a, b.T) / (np.dot(norm_a, norm_b.T) + 1e-6)

class Track:
    def __init__(self, detection, track_id, feature):
        self.id = track_id
        self.bbox = detection
        self.hits = 1
        self.time_since_update = 0
        self.feature = feature
        
        cx = (detection[0] + detection[2]) / 2
        cy = (detection[1] + detection[3]) / 2
        
        self.kf = KalmanFilter(dt=1, u_x=0, u_y=0, std_acc=0.1, x_std_meas=0.1, y_std_meas=0.1)
        self.kf.x = np.array([cx, cy, 0, 0])
        
        if feature is not None:
             self.feature = feature / (np.linalg.norm(feature) + 1e-6)
        else:
             self.feature = None

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
        
    def update(self, detection, feature=None):
        self.time_since_update = 0
        self.hits += 1
        
        cx = (detection[0] + detection[2]) / 2
        cy = (detection[1] + detection[3]) / 2
        self.kf.update(np.array([cx, cy]))
        
        if feature is not None:
             feature = feature / (np.linalg.norm(feature) + 1e-6)
             # Exponential Moving Average to update features smoothly
             if self.feature is None:
                 self.feature = feature
             else:
                 alpha = 0.9
                 self.feature = alpha * self.feature + (1 - alpha) * feature
                 self.feature = self.feature / (np.linalg.norm(self.feature) + 1e-6) 
        
        filtered_cx = self.kf.x[0]
        filtered_cy = self.kf.x[1]
        
        w = detection[2] - detection[0]
        h = detection[3] - detection[1]
        
        x1 = filtered_cx - w/2
        y1 = filtered_cy - h/2
        x2 = filtered_cx + w/2
        y2 = filtered_cy + h/2
        
        self.bbox = [x1, y1, x2, y2]


class AppearanceAwareTracker:
    def __init__(self, model_path, max_age=15, min_hits=1, iou_threshold=0.3, alpha=0.5, beta=0.5):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.frame_count = 0
        self.id_count = 1
        self.alpha = alpha
        self.beta = beta
        
        self.net = cv2.dnn.readNet(model_path)
        self.roi_width = 64
        self.roi_height = 128
        
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def preprocess_patch(self, im_crops):
        roi_input = cv2.resize(im_crops, (self.roi_width, self.roi_height))
        roi_input = cv2.cvtColor(roi_input, cv2.COLOR_BGR2RGB)
        roi_input = roi_input.astype(np.float32) / 255.0
        roi_input = (roi_input - self.mean) / self.std
        roi_input = np.transpose(roi_input, (2, 0, 1))
        roi_input = np.expand_dims(roi_input, axis=0)
        return roi_input
        
    def extract_features(self, frame, det_bboxes):
        features = []
        for det in det_bboxes:
            x1, y1, x2, y2 = det
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(frame.shape[1], int(x2))
            y2 = min(frame.shape[0], int(y2))
            
            if x2 <= x1 or y2 <= y1:
                features.append(np.zeros((512,), dtype=np.float32))
                continue
                
            patch = frame[y1:y2, x1:x2]
            blob = self.preprocess_patch(patch)
            self.net.setInput(blob)
            feature = self.net.forward()
            features.append(feature.flatten())
            
        return np.array(features)

    def update(self, frame, detections):
        self.frame_count += 1
        
        predicted_bboxes = []
        track_features = []
        for t in self.tracks:
            predicted_bboxes.append(t.predict())
            track_features.append(t.feature)
        
        predicted_bboxes = np.array(predicted_bboxes)
        track_features = np.array(track_features)
        
        if len(detections) > 0:
            det_bboxes = detections[:, :4]
            det_features = self.extract_features(frame, det_bboxes)
        else:
            det_bboxes = np.empty((0, 4))
            det_features = np.empty((0, 512))
            
        if len(predicted_bboxes) > 0 and len(det_bboxes) > 0:
            iou_matrix = iou_batch(det_bboxes, predicted_bboxes)
            
            sim_matrix = cosine_similarity(det_features, track_features)
            sim_matrix = np.maximum(0, sim_matrix)

            score_matrix = self.alpha * iou_matrix + self.beta * sim_matrix
            
            row_ind, col_ind = linear_sum_assignment(-score_matrix)
            matched_indices = np.stack((row_ind, col_ind), axis=1)
        else:
            matched_indices = np.empty((0,2))
            
        unmatched_detections = []
        for d in range(len(det_bboxes)):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
        
        unmatched_tracks = []
        for t in range(len(self.tracks)):
            if t not in matched_indices[:, 1]:
                unmatched_tracks.append(t)
                
        matches = []
        for m in matched_indices:
            det_idx = m[0]
            track_idx = m[1]
            # Improved matching logic:
            # Allow match if IoU is good OR if Appearance Similarity is very high.
            # This helps recover IDs after occlusion or fast motion where IoU drops.
            
            iou_score = iou_matrix[det_idx, track_idx]
            sim_score = sim_matrix[det_idx, track_idx]
            
            # Thresholds
            iou_pass = iou_score > self.iou_threshold
            sim_pass = sim_score > 0.75  # High confidence re-id parameters
            
            if iou_pass or sim_pass:
                 matches.append(m.reshape(1,2))
            else:
                 unmatched_detections.append(det_idx)
                 unmatched_tracks.append(track_idx)
        
        if len(matches) == 0:
            matches = np.empty((0,2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        for m in matches:
            det_idx = m[0]
            track_idx = m[1]
            self.tracks[track_idx].update(detections[det_idx][:4], det_features[det_idx])
            
        for i in unmatched_detections:
            self.tracks.append(Track(detections[i][:4], self.id_count, det_features[i]))
            self.id_count += 1
            
        for i in unmatched_tracks:
            self.tracks[i].time_since_update += 1
            
        for t in reversed(range(len(self.tracks))):
            if self.tracks[t].time_since_update > self.max_age:
                self.tracks.pop(t)
                
        ret = []
        for trk in self.tracks:
             if (trk.time_since_update < 1) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits):
                 ret.append(np.concatenate((trk.bbox, [trk.id]))) 
        return ret

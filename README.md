# Object Tracking Project

This project explores different techniques for Object Tracking, progressing from basic single-object tracking to advanced multi-object tracking with re-identification.

## Content

The project is divided into four main parts (TPs):

- **TP1: Single Object Tracking**
  - Implements a centroid tracker using **Kalman Filter**.
  - tracks a single object (black dot).

- **TP2: Multi-Object Tracking (MOT)**
  - Implements an **IoU-based tracker**.
  - Handles multiple objects but suffers from ID switching during occlusions.

- **TP3: Kalman + IoU Tracking**
  - improved robustness by combining **Kalman Filter** path prediction with **IoU** association.
  - Reduces ID switching but still struggles with complex occlusions (swapping IDs).

- **TP4: Appearance-Aware Tracking**
  - Integrates a **Re-Identification (ReID)** model.
  - Uses visual features to handle long-term occlusions and maintain stable IDs.

## Report

A detailed analysis of the methods and results can be found in the `report/` directory:
- [report.pdf](report/report.pdf)

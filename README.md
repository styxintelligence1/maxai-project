# Camera-based AI Training System

Recreated project skeleton after container wipe.

## Key directories

- data/raw/gopro           – Raw GoPro / 360 footage
- data/raw/fixed_cams      – Raw fixed-camera footage
- data/frames              – Extracted frames from videos
- data/labels/cvat_exports – CVAT export zips / directories
- data/labels/yolo         – Final YOLO-style labels
- data/datasets            – Train/val/test dataset folders
- models/yolo              – YOLO runs and weights
- scripts/pipeline         – Training pipeline scripts
- scripts/cvat             – CVAT → YOLO conversion / helpers
- config                   – YAML configs for paths, YOLO, etc.
- docker                   – Dockerfile and Compose files

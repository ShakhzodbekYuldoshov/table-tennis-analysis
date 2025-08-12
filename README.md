# Table Tennis Analysis

### Computer Vision Pipeline
- Frame differencing for motion detection
- Player filtering using YOLO person detection
- Morphological operations for noise reduction
- Contour-based object detection

## Installation
### Prerequisites
1. nvidia driver
2. cuda toolkit
3. cudnn

Then we can install other requirements
```
pip3 install -r requirements.txt
```

## Usage

### Tracking with imshow
```
python3 ball_tracking_demo.py --source ./videos/IMG_7370.MOV
```

### Tracking with imshow
```
python3 ball_tracking_demo.py --source ./videos/IMG_7370.MOV --output ./out.mp4
```

## Project Structure

```
├── ball_tracking_demo.py   # Tracking demo
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── models/
│   ├── cv_technique.py    # Computer vision detection with tracking
│   ├── yolo11.py         # YOLO v11 integration
│   └── yolov5.py         # YOLO v5 integration
├── datatypes/
│   └── track_object.py   # Tracking classes and algorithms
├── weights/              # Model weights
├── videos/               # Input/output videos
└── BALL_TRACKING.md      # Detailed tracking documentation
```
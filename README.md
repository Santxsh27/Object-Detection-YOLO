# Object Detection using YOLOv3 and YOLOv4 with GPU vs Without GPU

This project demonstrates the implementation of object detection using the YOLOv3 and YOLOv4 models with a comparison of performance in terms of FPS (frames per second) with and without GPU acceleration. The system detects objects in real-time through the webcam and provides an estimation of object distance and size.

## Table of Contents
- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Running the Code](#running-the-code)
  - [Using YOLOv3](#using-yolov3)
  - [Using YOLOv4](#using-yolov4)
- [Explanation of Functions](#explanation-of-functions)
- [Comparing GPU vs Without GPU](#comparing-gpu-vs-without-gpu)
- [Results](#results)
- [License](#license)

## Introduction

The purpose of this project is to explore the differences in performance between using a GPU and not using a GPU for object detection in real-time. This is achieved using YOLOv3 and YOLOv4 models. YOLO (You Only Look Once) is a popular object detection algorithm that is capable of processing images in real-time. We will display the FPS to show how GPU acceleration impacts the speed.

## Prerequisites

Before you can run this project, ensure you have the following installed:

- Python 3.x
- OpenCV (with CUDA enabled for GPU acceleration)
- Numpy
- YOLOv3 and YOLOv4 model files (weights, config, and coco.names)

## Installation

1. **Clone this repository:**

   ```bash
   git clone https://github.com/your-username/Object-Detection-YOLO.git
   cd Object-Detection-YOLO

   Install the required Python libraries:

2.Make sure you are in the project directory, then install the dependencies:

pip install opencv-python opencv-python-headless numpy

# Download YOLO model files:

Ensure you have the following YOLOv3 and YOLOv4 files in the project folder:

yolov3.cfg
yolov3.weights
yolov4.cfg
yolov4.weights
coco.names
If you don’t have these files, download them from the official YOLO website.

# Project Structure
bash
Copy code
Object-Detection-YOLO/
│
├── objectdetection.py      # Main Python script for running the project
├── coco.names              # COCO dataset class names
├── yolov3.cfg              # YOLOv3 model configuration file
├── yolov3.weights          # YOLOv3 pre-trained weights
├── yolov4.cfg              # YOLOv4 model configuration file
├── yolov4.weights          # YOLOv4 pre-trained weights
├── test/                   # Test data folder (if required)
└── train/                  # Training data folder (if required)
Running the Code
Run the object detection code:

After navigating to the project directory, run the objectdetection.py script:

bash
Copy code
python objectdetection.py
The script will prompt you to select the YOLO model you wish to use (YOLOv3 or YOLOv4).

# Select the YOLO model:

Upon running the code, the terminal will display the following options:

diff
Copy code
Available models:
- yolov3
- yolov4
Type the model name you'd like to use (e.g., yolov3) and hit enter. The webcam will open, and real-time object detection will start.

Press q to quit: At any point, you can press the q key to exit the webcam window.

# Using YOLOv3
When you select yolov3 as your model, the YOLOv3 weights and configuration will be loaded for object detection.

# Using YOLOv4
Similarly, selecting yolov4 will use the YOLOv4 configuration and weights.

# Explanation of Functions
load_model(model_name)
Loads the specified YOLO model's weights and configuration.
Returns the pre-trained network and the class names from the COCO dataset.
detect_objects(frame, net, classes)
Processes each frame using the selected YOLO model.
Applies GPU acceleration if available.
Outputs the bounding boxes, confidence scores, and class labels for detected objects.
Calculates object size and distance based on the known width of the object and the camera's focal length.
estimate_distance(object_width, known_width, focal_length)
Estimates the distance between the camera and the object using the object's detected width in pixels, the known width in centimeters, and the camera's focal length.
calculate_object_size(object_width, object_height, known_width, distance, focal_length)
Estimates the object's size based on its detected dimensions, the camera's distance, and the focal length.
Comparing GPU vs Without GPU
This project uses OpenCV’s DNN_BACKEND_CUDA to enable GPU acceleration. When running the code, the FPS (frames per second) with and without GPU is displayed at the top left of the screen:

FPS (No GPU): Measures the frames per second without GPU acceleration.
FPS (With GPU): Measures the frames per second with GPU acceleration.
The difference in FPS gives an insight into how much faster GPU acceleration makes the object detection process.

# Results
YOLOv3:
FPS without GPU: ~2-4 FPS (depending on hardware)
FPS with GPU: ~15-30 FPS (depending on hardware)
YOLOv4:
FPS without GPU: ~2-4 FPS (depending on hardware)
FPS with GPU: ~20-40 FPS (depending on hardware)
The FPS difference is notable and highlights the efficiency of utilizing GPU for object detection tasks.



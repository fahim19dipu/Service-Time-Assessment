# Service Time Assessment - YOLO-based Object Tracking

This project tracks customers in a checkout area of a store from a video feed, calculates the time each customer spends in the checkout area (service time), and provides summary statistics such as total and average service time.

## Table of Contents

1. [Installation](#installation)
2. [Requirements](#requirements)
3. [How It Works](#how-it-works)
4. [Usage](#usage)
5. [License](#license)

## Installation

### Prerequisites

You need Python installed on your machine along with the following libraries:

- OpenCV (`cv2`)
- NumPy
- `ultralytics` (for YOLO model)

To install the required dependencies, create a virtual environment and install the libraries using `pip`:

```bash
# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install opencv-python numpy ultralytics
```
### YOLO Model
This script uses a pre-trained YOLO model to track the objects. Make sure to have the correct YOLO model file (yolo11n.pt) for object detection. You can download it from a repository or replace it with another YOLO model.
### Requirements
1. Python 3.x
2. OpenCV (cv2)
3. NumPy
4. ultralytics library (for YOLO tracking)
5. A video file (e.g., fringestorez.mp4) for testing.

### How It Works
#### Select Region of Interest (ROI):
The user is prompted to select the checkout area (ROI) in the first frame of the video. The program then tracks the objects (customers) inside this region.

#### Object Tracking with YOLO:
The YOLO model is used to detect and track objects in the video. The script assigns unique IDs to the objects and stores their position in the frame.

#### Tracking Entry and Exit Times:
The script monitors when customers enter and exit the checkout area, logging their entry and exit times.

#### Calculating Service Time:
For each customer, the service time is calculated as the difference between their entry and exit times. Customers who spend less than 2 seconds in the checkout area are excluded to avoid outliers.

### Results:
After processing the video, the script displays:
<ul>
<li>The service time for each customer.</li>
<li>Total customers served.</li>
<li>Total service time.</li>
<li>Average service time per customer.</li>
</ul>

### Usage
1. Prepare the Video: Make sure the video file (e.g., test.mp4) is in the same directory as the script or update the video_path in the script.
2. Run the Script:
```bash
python service_time.py
```
3. Select ROI: The program will open the first frame of the video and ask you to select the region that corresponds to the checkout area using a drag-to-select box. Press Enter once the area is selected.

4. View Progress: The script will process the video frame by frame. It will show the annotated frames with tracked customers. Press q to stop the video processing and display results.

5. Results: After processing the video, the service times for each customer will be printed in the console, along with the total and average service times.

# -*- coding: utf-8 -*-
"""
Created on Sun Dec  15 11:54:28 2024

@author: Fahim Abdullah
"""

from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import logging
logging.getLogger("ultralytics").setLevel(logging.WARNING)
# Function to select ROI
def select_roi(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Failed to load video!")
        return None, None

    # Add instruction text on the frame
    instruction_text = "*Instruction*:- Please select the checkout area and press Enter"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_color = (0, 255, 0)  # Green
    thickness = 2
    org = (10, 30)  # Top-left corner

    # Overlay the text on the frame
    cv2.putText(frame, instruction_text, org, font, font_scale, font_color, thickness, cv2.LINE_AA)

    # Select ROI manually
    roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    return roi, frame

# Load the YOLO model
model = YOLO("yolo11n.pt")

# Open the video file
video_path = "fringestorez.mp4"

# Get ROI from the user
bbox, frame = select_roi(video_path)
if bbox:
    x1, y1, w, h = bbox
    x2 = x1 + w
    y2 = y1 + h
    checkout_area_x = (x1, x2)
    checkout_area_y = (y1, y2)
    print(f"Selected ROI for checkout: X={checkout_area_x}, Y={checkout_area_y}")
else:
    print("No ROI selected. Exiting.")
    exit()

# # Store the track history
track_history = defaultdict(lambda: [])
entry_times = defaultdict(lambda: None)
exit_times = defaultdict(lambda: None)

# Function to check if a customer is in the checkout area
def in_checkout_area(x, y):
    return checkout_area_x[0] <= x <= checkout_area_x[1] and checkout_area_y[0] <= y <= checkout_area_y[1]

# # Open the video file for processing
cap = cv2.VideoCapture(video_path)

# # Total frame count (used as a fail-safe)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
current_frame = 0

# # Loop through the video frames

# Retrieve frame rate from the video
fps = cap.get(cv2.CAP_PROP_FPS)
frame_duration = 1 / fps  # Duration of each frame in seconds

# Updated entry and exit tracking
entry_frames = defaultdict(lambda: None)
exit_frames = defaultdict(lambda: None)
print("Please Wait while your video is being processed")
# Loop through video frames
while cap.isOpened():
    print('*', flush=True, end='\r')
    success, frame = cap.read()
    current_frame += 1

    if not success:
        print("End of video or failed to read frame. Exiting loop.")
        break

    # Run YOLO tracking
    results = model.track(frame, persist=True)

    # Get detections
    boxes = results[0].boxes.xywh.cpu()
    track_ids = results[0].boxes.id.int().cpu().tolist()

    # Visualize results
    annotated_frame = results[0].plot()

    # Loop through detections
    for box, track_id in zip(boxes, track_ids):
        x, y, w, h = box
        track = track_history[track_id]
        track.append((float(x), float(y)))

        if len(track) > 30:
            track.pop(0)

        # Check if the customer enters or leaves the checkout area
        if in_checkout_area(x, y):
            if entry_frames[track_id] is None:  # First time entering
                entry_frames[track_id] = current_frame  # Record frame number
        elif entry_frames[track_id] is not None and exit_frames[track_id] is None:
            exit_frames[track_id] = current_frame  # Record frame number

        # Draw tracking lines
        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

    
    # Add instruction text on the frame
    instruction_text = "*Instruction*:- Press q to stop processing and Show result"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_color = (0, 255, 0)  # Green
    thickness = 2
    org = (10, 30)  # Top-left corner

    # Overlay the text on the frame
    cv2.putText(annotated_frame, instruction_text, org, font, font_scale, font_color, thickness, cv2.LINE_AA)

    # Display annotated frame
    cv2.imshow("Service Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("User requested to exit. Exiting loop.")
        break

    if current_frame >= total_frames:
        print("Processed all frames. Exiting loop.")
        break
    print(' ', flush=True, end='\r')
    
total_time = 0
served_count = 0

print("Service times for each customer:")
for track_id in entry_frames:
    if entry_frames[track_id] is not None and exit_frames[track_id] is not None:
        # Calculate service time for this customer
        service_time = (exit_frames[track_id] - entry_frames[track_id]) * frame_duration

        # Skip customers with service time less than 2 seconds to reduce outliers and misclaissified results 
        if service_time < 2:
            continue

        print(f"Customer {track_id}: {service_time:.2f} seconds")
        total_time += service_time
        served_count += 1

# Calculate and display the average service time
if served_count > 0:
    average_time = total_time / served_count
    print("\nSummary:")
    print(f"Total customers served: {served_count}")
    print(f"Total service time: {total_time:.2f} seconds")
    print(f"Average service time: {average_time:.2f} seconds")
else:
    print("No customers were served.")
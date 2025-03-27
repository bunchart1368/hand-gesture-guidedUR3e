"""
Auto Gesture Labeler
This script automatically labels hand gestures in a video file using a trained model.
It processes each frame and saves the gesture labels to a CSV file.
"""

import csv
import sys
import os
import cv2
import numpy as np
import mediapipe as mp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.config import settings
from model import KeyPointClassifier  # Import your classifier class
from function import calc_bounding_rect, calc_landmark_list, pre_process_landmark
from model.keypoint_classifier.transformer import MaskFeatureSelector


def load_gesture_names(gesture_names_path):
    """Load gesture names from csv file"""
    with open(gesture_names_path, encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
        return keypoint_classifier_labels

def save_prediction(prediction, output_path):
    """Save ground truth labels to CSV file with separate columns for left and right hands"""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_idx', 'left_hand_gesture', 'right_hand_gesture'])
        for frame_idx, data in sorted(prediction.items()):
            left_gesture = data.get('left', None)  # Use None if left hand not detected
            right_gesture = data.get('right', None)  # Use None if right hand not detected
            writer.writerow([frame_idx, left_gesture, right_gesture])

def main():
    # Load paths from settings
    video_path = settings.model_evaluation.video_path
    output_path = settings.model_evaluation.output_path
    gesture_names_path = settings.model_evaluation.gesture_names_path
    frame_step = settings.model_evaluation.frame_step
    model_path = settings.keypoint_classifier.model_path  # Add this to your settings
    
    # Load gesture names
    gesture_names = load_gesture_names(gesture_names_path)
    print("Available gestures:", gesture_names)
    
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Initialize keypoint classifier
    keypoint_classifier = KeyPointClassifier(model_path=model_path)
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    prediction = {}
    frame_idx = 0
    
    print(f"Processing video: {video_path}")
    print(f"Total frames: {frame_count}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_step == 0:
            # Create entry for this frame
            if frame_idx not in prediction:
                prediction[frame_idx] = {}
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame with MediaPipe Hands
            results = hands.process(frame_rgb)
            
            # Default values (None) if no hands are detected
            hand_sign_id_left = None
            hand_sign_id_right = None
            
            # If hands are detected, process them
            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # Bounding box calculation
                    brect = calc_bounding_rect(frame, hand_landmarks)
                    
                    # Landmark calculation
                    landmark_list = calc_landmark_list(frame, hand_landmarks)
                    
                    # Conversion to relative coordinates / normalized coordinates
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)
                    
                    # Hand sign classification
                    hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                    
                    # Assign hand sign based on handedness
                    if handedness.classification[0].label == "Right":
                        hand_sign_id_right = hand_sign_id
                    if handedness.classification[0].label == "Left":
                        hand_sign_id_left = hand_sign_id
            
            # Save the detected gesture IDs
            prediction[frame_idx]['left'] = hand_sign_id_left
            prediction[frame_idx]['right'] = hand_sign_id_right
            
            # Display progress
            print(f"Frame {frame_idx}/{frame_count}: Left hand: {hand_sign_id_left}, Right hand: {hand_sign_id_right}")
            
            # Optional: Display the frame with results
            if True:
                # Draw results on frame (implement as needed)
                cv2.imshow('Frame', frame)
                key = cv2.waitKey(150)
                if key == 27:  # ESC key
                    break
        
        frame_idx += 1
        
        # Show progress
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{frame_count} frames")
    
    # Save ground truth
    save_prediction(prediction, output_path)
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"Automatic gesture labeling complete!")
    print(f"Ground truth saved to {output_path}")

if __name__ == '__main__':
    main()
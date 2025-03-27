"""
Ground Truth Generator
This script helps to create ground truth labels for a video file.
It shows each frame of the video and lets you assign a gesture label.
"""

import argparse
import csv
import sys
import json
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.config import settings
import cv2

def load_gesture_names(gesture_names_path):
    """Load gesture names from csv file"""
    with open(gesture_names_path, encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
        return keypoint_classifier_labels

def save_ground_truth(ground_truth, output_path):
    """Save ground truth labels to CSV file with separate columns for left and right hands"""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_idx', 'left_hand_gesture', 'right_hand_gesture'])
        for frame_idx, data in sorted(ground_truth.items()):
            left_gesture = data.get('left', None)  # Use None if left hand not detected
            right_gesture = data.get('right', None)  # Use None if right hand not detected
            writer.writerow([frame_idx, left_gesture, right_gesture])

def main():
    video_path = settings.model_evaluation.video_path
    output_path = settings.model_evaluation.prediction_output_path
    gesture_names_path = settings.model_evaluation.gesture_names_path
    frame_step = settings.model_evaluation.frame_step
    
    # Load gesture names
    gesture_names = load_gesture_names(gesture_names_path)
    
    # Print available gestures
    print("Available gestures:", gesture_names)
    # for label, name in gesture_names.items():
    #     print(f"{label}: {name}")
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    ground_truth = {}
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_step == 0:
            # Display frame
            cv2.imshow('Frame', frame)
            cv2.waitKey(1)
            
            # Display frame information
            print(f"\nFrame {frame_idx}/{frame_count}")

            # Create entry for this frame
            if frame_idx not in ground_truth:
                ground_truth[frame_idx] = {}
            
            # Prompt for gesture label
            while True:
                try:
                    left_label = input("Enter LEFT hand gesture (or 'n' for none, 's' to skip frame, 'q' to quit): ")
                    if left_label == 'q':
                        save_ground_truth(ground_truth, output_path)
                        cap.release()
                        cv2.destroyAllWindows()
                        print(f"Ground truth saved to {output_path}")
                        return
                    elif left_label == 's':
                        break
                    elif left_label == 'n':
                        # No left hand detected - explicitly set to None
                        ground_truth[frame_idx]['left'] = None
                        break
                    else:
                        left_label = int(left_label)
                        if 0 <= left_label < len(gesture_names):
                            ground_truth[frame_idx]['left'] = left_label
                            print(f"Left hand labeled as {gesture_names[left_label]}")
                            break
                        else:
                            print(f"Invalid label. Please enter a number from 0 to {len(gesture_names)-1}")
                except ValueError:
                    print("Please enter a valid number")

            # Skip right hand input if user wants to skip the frame
            if left_label != 's':
                # Ask for right hand gesture
                while True:
                    try:
                        right_label = input("Enter RIGHT hand gesture (or 'n' for none, 'q' to quit): ")
                        if right_label == 'q':
                            save_ground_truth(ground_truth, output_path)
                            cap.release()
                            cv2.destroyAllWindows()
                            print(f"Ground truth saved to {output_path}")
                            return
                        elif right_label == 'n':
                            # No right hand detected - explicitly set to None
                            ground_truth[frame_idx]['right'] = None
                            break
                        else:
                            right_label = int(right_label)
                            if 0 <= right_label < len(gesture_names):
                                ground_truth[frame_idx]['right'] = right_label
                                print(f"Right hand labeled as {gesture_names[right_label]}")
                                break
                            else:
                                print(f"Invalid label. Please enter a number from 0 to {len(gesture_names)-1}")
                    except ValueError:
                        print("Please enter a valid number")
                        # Wait for key press
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
        
        frame_idx += 1
        
        # Show progress
        if frame_idx % 100 == 0:
            print(f"Processing frame {frame_idx}/{frame_count}")
                
    # Save ground truth
    save_ground_truth(ground_truth, output_path)
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"Ground truth saved to {output_path}")

if __name__ == '__main__':
    main()
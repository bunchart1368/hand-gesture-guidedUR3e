"""
Ground Truth Generator
This script helps to create ground truth labels for a video file.
It shows each frame of the video and lets you assign a gesture label.
"""

import argparse
import csv
import sys
import json
import copy
import os
import cv2 as cv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.config import settings
import cv2
import mediapipe as mp
from function import calc_bounding_rect, calc_landmark_list, pre_process_landmark, draw_bounding_rect, draw_landmarks


def load_gesture_names(gesture_names_path):
    """Load gesture names from csv file"""
    with open(gesture_names_path, encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
        return keypoint_classifier_labels

def save_ground_truth(ground_truth, output_path):
    with open(output_path, 'w', newline='') as f:
        # Create column headers
        headers = ['frame_idx', 'handedness']
        
        # Add headers for each landmark (21 landmarks with x,y coordinates)
        for i in range(21):
            headers.extend([f'lm{i}_x', f'lm{i}_y'])
        
        # Add gesture label column
        headers.append('gesture')
        
        writer = csv.writer(f)
        writer.writerow(headers)
        
        # Write data rows
        for frame_idx, data in sorted(ground_truth.items()):
            # Process left hand if present
            if 'left_lm' in data and data['left_lm'] is not None:
                left_row = [frame_idx, 'left']
                
                # Extract the 42 values (21 landmarks × 2 coordinates) into separate columns
                landmarks = data['left_lm']
                for i in range(0, len(landmarks), 2):
                    if i + 1 < len(landmarks):  # Ensure we have both x and y
                        left_row.append(landmarks[i])      # x-coordinate
                        left_row.append(landmarks[i + 1])  # y-coordinate
                
                # Add gesture label (if available, otherwise None)
                left_label = data.get('left_label', None)
                left_row.append(left_label)
                
                writer.writerow(left_row)
            
            # Process right hand if present
            if 'right_lm' in data and data['right_lm'] is not None:
                right_row = [frame_idx, 'right']
                
                # Extract the 42 values (21 landmarks × 2 coordinates) into separate columns
                landmarks = data['right_lm']
                for i in range(0, len(landmarks), 2):
                    if i + 1 < len(landmarks):  # Ensure we have both x and y
                        right_row.append(landmarks[i])      # x-coordinate
                        right_row.append(landmarks[i + 1])  # y-coordinate
                
                # Add gesture label (if available, otherwise None)
                right_label = data.get('right_label', None)
                right_row.append(right_label)
                
                writer.writerow(right_row)

def main():
    video_path = settings.model_evaluation.video_path
    output_path = settings.model_evaluation.groundtruth_output_path
    gesture_names_path = settings.model_evaluation.gesture_names_path
    frame_step = settings.model_evaluation.frame_step

    print('vdo source:', video_path)
    print('output path:', output_path)
    
    # Load gesture names
    gesture_names = load_gesture_names(gesture_names_path)
    
    # Print available gestures
    print("Available gestures:", gesture_names)
    # for label, name in gesture_names.items():
    #     print(f"{label}: {name}")

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    ground_truth = {}
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # frame = cv.flip(frame, 1)  # Mirror display
        debug_image = copy.deepcopy(frame)
        
        if frame_idx % frame_step == 0:
            # Display frame
            cv2.imshow('Frame', frame)
            cv2.waitKey(1)
            
            # Display frame information
            print(f"\nFrame {frame_idx}/{frame_count}")

            # Create entry for this frame
            if frame_idx not in ground_truth:
                ground_truth[frame_idx] = {}

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame with MediaPipe Hands
            # frame_rgb = cv.flip(frame_rgb, 1)
            results = hands.process(frame_rgb)

            # If hands are detected, process them
            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    
                    # Bounding box calculation
                    brect = calc_bounding_rect(debug_image, hand_landmarks)

                    # Landmark calculation
                    landmark_list = calc_landmark_list(frame, hand_landmarks)
                    
                    # Conversion to relative coordinates / normalized coordinates
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)
                    handedness= handedness.classification[0].label
                    print('handedness', handedness)
                    # print('hand landmarks:', hand_landmarks)
                    
                    # Store landmark data
                    print('enter condision...')
                    if handedness == "Right":
                        ground_truth[frame_idx]['right_lm'] = pre_processed_landmark_list
                        print(f"Right hand landmarks: {pre_processed_landmark_list[:4]}")
                    if handedness == "Left":
                        ground_truth[frame_idx]['left_lm'] = pre_processed_landmark_list
                        print(f"Left hand landmarks: {pre_processed_landmark_list[:4]}")

                    
                    debug_image = draw_bounding_rect(True, debug_image, brect)
                    debug_image = draw_landmarks(debug_image, landmark_list)

                    # Add text to indicate which hand
                    org = (brect[0], brect[1] - 10)
                    cv2.putText(debug_image, f"{handedness} Hand, {pre_processed_landmark_list[2]} landmark", org, 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                           
            cv.imshow('Hand Gesture Recognition', debug_image)
            cv2.waitKey(1)
            
            
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
                    elif left_label == '':
                        break
                    elif left_label == 'n':
                        # No left hand detected - explicitly set to None
                        ground_truth[frame_idx]['left_label'] = None
                        break
                    else:
                        left_label = int(left_label)
                        if 0 <= left_label < len(gesture_names):
                            ground_truth[frame_idx]['left_label'] = left_label
                            print(f"Left hand labeled as {gesture_names[left_label]}")
                            break
                        else:
                            print(f"Invalid label. Please enter a number from 0 to {len(gesture_names)-1}")
                except ValueError:
                    print("Please enter a valid number")

            # Skip right hand input if user wants to skip the frame
            if left_label != '':
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
                            ground_truth[frame_idx]['right_label'] = None
                            break
                        else:
                            right_label = int(right_label)
                            if 0 <= right_label < len(gesture_names):
                                ground_truth[frame_idx]['right_label'] = right_label
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
        # print('frame_idx:', frame_idx)
        # print('ground_truth:', ground_truth)
        
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
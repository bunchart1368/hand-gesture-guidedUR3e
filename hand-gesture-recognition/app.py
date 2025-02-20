#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import socket
import itertools
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier

from function import ( select_mode, calc_bounding_rect, calc_landmark_list, 
                      pre_process_landmark, pre_process_point_history, logging_csv, draw_bounding_rect, 
                      draw_landmarks, draw_info_text, draw_point_history, draw_info )

from utils.config import settings

# Server connection
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('localhost', 12345))
server_socket.listen(1)
print("Server is waiting for a connection...")

conn, addr = server_socket.accept()
print(f"Connected to {addr}")

# model path hand gesture
model_path = settings.keypoint_classifier.model_path
model_label_path = settings.keypoint_classifier.label_path

# model path finger gesture
model_point_path = settings.point_history_classifier.model_path
model_point_label_path = settings.point_history_classifier.label_path

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


def main():
    # Argument parsing #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    point_history_classifier = PointHistoryClassifier()

    # Read labels ###########################################################
    with open(model_label_path,
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
            model_point_label_path,
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history #################################################################
    history_length = 16
    point_history = deque(maxlen=history_length)

    # Finger gesture history ################################################
    finger_gesture_history = deque(maxlen=history_length)

    #  ########################################################################
    mode = 0

    while True:
        fps = cvFpsCalc.get()

        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)     # create hand landmark
        image.flags.writeable = True
        # print('results', results.multi_hand_landmarks)

        hand_sign_id_right = 0
        hand_sign_id_left = 0
        
        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                print('handedness', handedness)
                print('hand landmarks',hand_landmarks)
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                # print('landmark list', landmark_list)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                # print('')
                # print('pre-processed landmark list', pre_processed_landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history)
                # Write to the dataset file
                logging_csv(number, mode, pre_processed_landmark_list,
                            pre_processed_point_history_list)

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

                # Assign hand sign based on handedness
                if handedness.classification[0].label == "Right":
                    print('-----a',handedness.classification[0].label)
                    hand_sign_id_right = hand_sign_id
                if handedness.classification[0].label == "Left":
                    print('-----b',handedness.classification[0].label)
                    hand_sign_id_left = hand_sign_id

                print('hand_sign_id_right', hand_sign_id_right)
                print('hand_sign_id_left', hand_sign_id_left)
                

                        # Gesture-based commands
                if hand_sign_id_right == 1 and hand_sign_id_left == 0 :  # Right hand Scissors gesture
                    command = 'Right' # Right 
                    command_id = 0
                elif hand_sign_id_right == 0 and hand_sign_id_left == 1:  # Left hand Point gesture
                    command = 'Left' # Left
                    command_id = 1
                    point_history.append(landmark_list[8])
                elif hand_sign_id_right == 1 and hand_sign_id_left == 1:
                    command = 'Up' # Up
                    command_id = 2
                elif hand_sign_id_right == 0 and hand_sign_id_left == 0:
                    command = 'Down' # Down
                    command_id = 3
                    point_history.append([0, 0])

                print('command', command)

                # # Finger gesture classification
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list)

                # # Calculates the gesture IDs in the latest detection
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common()

                # Drawing part
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)

                # Draw angle
                # debug_image = draw_finger_angles(debug_image, results, command) 
                debug_image, magnitude_angle = draw_angles_command(debug_image, results, command)       
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                )
                #send command to robot
                send_command(command, magnitude_angle)
        else:
            point_history.append([0, 0])

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)
        # Screen reflection #############################################################
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()


def draw_angel_lines(image, a, b, c):
    a_pos = tuple(np.multiply(a, [640, 480]).astype(int))
    b_pos = tuple(np.multiply(b, [640, 480]).astype(int))
    c_pos = tuple(np.multiply(c, [640, 480]).astype(int))
    
    cv.line(image, a_pos, b_pos, (0, 255, 0), 2)
    cv.line(image, b_pos, c_pos, (0, 255, 0), 2)
    
    return image

def draw_angles_command(image, results, command):
    # Loop through hands
    joint_dict = {
                    # 'zoom': [8, 2, 4], 
                    # 'rotate' : [4, 0, 17], 
                    # 'tilt (LR)': [4, 0, 17], 
                    'Up': [15,18,20],
                    'Down': [15,18,20],
                    'Left': [15,18,20],
                    'Right': [15,18,20],
                }
    for hand in results.multi_hand_landmarks:
        #Loop through joint sets 
        joint = joint_dict[command]
        a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y]) # First coord
        b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y]) # Second coord
        c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y]) # Third coord
            
        radians = np.arctan2(c[1] - b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
            
        if angle > 180.0:
            angle = 360-angle

        image = draw_angel_lines(image, a, b, c)

        cv.putText(image, command      +str(round(angle, 2)), tuple(np.multiply(b, [640, 480]).astype(int)),
                    cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv.LINE_AA)

    return image, angle

def send_command(command_id, angle):
    info_to_send = f"({command_id},{angle})"
    print(info_to_send)
    conn.send(info_to_send.encode())

if __name__ == '__main__':
    main()

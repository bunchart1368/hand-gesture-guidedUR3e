"""
Face_tracking01 with TCP Pivoting Control
Python program for realtime face tracking of a Universal Robot (tested with UR5cb)
Demonstration Video: https://youtu.be/HHb-5dZoPFQ
Explanation Video: https://www.youtube.com/watch?v=9XCNE0BmtUg

Created by Robin Godwyll (modified to include TCP pivoting control)
License: GPL v3 https://www.gnu.org/licenses/gpl-3.0.en.html
"""

# Standard library imports
import math
import sys
import time
import re
import socket
from typing import Tuple, List, Dict

# Third-party library imports
import numpy as np
import cv2
import imutils
from imutils.video import VideoStream
import math3d as m3d

# Project-specific imports
import URBasic
import keyboard
import yaml

"""SETTINGS AND VARIABLES ________________________________________________________________"""

RASPBERRY_BOOL = False

# Load YAML configuration directly
with open("./final-project-1/robot_variables.yml", "r") as file:
    config = yaml.safe_load(file)

ROBOT_IP = config["connection"]["ip_address"]
ACCELERATION = config["acceleration"]["joint_acceleration"]  # Robot acceleration value
VELOCITY = config["speed"]["joint_speed"]  # Robot speed value

# The Joint position the robot starts at
robot_startposition = [round(math.radians(degree), 3) for degree in config["initial_position"]["joint_angles"]]

# Size of the robot view-window
max_up = config["end_effector_limits"]["max_up"]
max_down = config["end_effector_limits"]["max_down"]
max_left = config["end_effector_limits"]["max_left"]
max_right = config["end_effector_limits"]["max_right"]

# Maximum Rotation of the robot at the edge of the view window
hor_rot_max = math.radians(45)
vert_rot_max = math.radians(45)

# Emergency stop flag
emergency_stop = False

# Global variables for TCP velocity estimation
prev_tcp = None  # Previous TCP position (list)
prev_time = None  # Time of previous update

"""FUNCTIONS _____________________________________________________________________________"""

def set_lookorigin() -> m3d.Transform:
    """
    Creates a new coordinate system at the current robot TCP position.
    This coordinate system is used as the reference for face tracking.
    """
    position = robot.get_actual_tcp_pose()
    orig = m3d.Transform(position)
    return orig

def extract_coordinates_from_orientation(oriented_xyz: m3d.Transform) -> List[float]:
    """
    Extracts numerical coordinates from a math3d Transform object's pose vector.
    """
    oriented_xyz_coord = oriented_xyz.pose_vector
    coordinates_str = str(oriented_xyz_coord)
    numbers = re.findall(r"-?\d+\.\d+", coordinates_str)
    coordinates = [float(num) for num in numbers]
    return coordinates

def detect_sign_change(previous: float, current: float) -> bool:
    return (previous > 0 and current < 0) or (previous < 0 and current > 0)

def server_connection():
    global client_socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('localhost', 12345))
    print("Connected to the server.")

def get_from_server() -> str:
    data = client_socket.recv(1024).decode()
    return data

def robot_set_up():
    global robot, robotModel
    print("Initialising robot")
    robotModel = URBasic.robotModel.RobotModel()
    robot = URBasic.urScriptExt.UrScriptExt(host=ROBOT_IP, robotModel=robotModel)
    print("RobotModel initialised")
    robot.reset_error()
    print("Robot initialised")
    time.sleep(1)

def check_boundary(accumulated_pose: List[float], target_pose: List[float]):
    temp_accumulated_pose = [sum(pair) + accumulated_pose[i] for i, pair in enumerate(zip(target_pose[::2], target_pose[1::2]))]
    for i, pose in enumerate(temp_accumulated_pose):
        if i == 0 and (pose > max_right or pose < max_left):
            target_pose[:] = [0] * len(target_pose)
            print("Reached max right or max left")
        if i == 1 and (pose > max_up or pose < max_down):
            target_pose[:] = [0] * len(target_pose)
            print("Reached max up or max down")
    accumulated_pose = [sum(pair) + accumulated_pose[i] for i, pair in enumerate(zip(target_pose[::2], target_pose[1::2]))]
    return accumulated_pose, target_pose

def home():
    robot.movej(q=robot_startposition, a=ACCELERATION, v=VELOCITY)
    print("Set home")

def set_new_tcp(offset: float):
    coordinates = [0, 0, offset, 0, 0, 0]
    print("NEW TCP Coordinates: ", coordinates)
    robot.set_tcp(coordinates)

def compute_target_pose(
    prev_pose: List[float],
    position: float,
    command: int,
    prev_ampli: float
) -> Tuple[List[float], float]:
    """
    Computes the target pose for the robot based on the previous pose and the current position input.
    """
    target_pose = prev_pose[:]
    scale_factor = 0.005
    position = int(position)
    if command == 1:
        position = 10
        target_pose[0] = int(position) * scale_factor
    elif command == 2:
        position = -10
        target_pose[1] = int(position) * scale_factor
    elif command == 3:
        position = 10
        target_pose[2] = int(position) * scale_factor
    elif command == 4:
        position = -10
        target_pose[3] = int(position) * scale_factor
    else:
        position = 0
        target_pose[:] = [0] * len(target_pose)

    prev_ampli = position
    return target_pose, prev_ampli

def compute_tcp_pivot_correction(current_tcp: List[float],
                                 fulcrum: List[float],
                                 previous_tcp: List[float],
                                 dt: float,
                                 kAx: float,
                                 kAy: float,
                                 F_Bx: float,
                                 F_By: float) -> List[float]:
    """
    Compute the angular correction (in rad/s) to be added to the TCP orientation.
    Uses an admittance term (proportional to the force) and a feedforward term 
    (predicting the future angle change based on the TCP velocity).
    
    Parameters:
        current_tcp: Current TCP position as [x, y, z].
        fulcrum: The fulcrum point (reference) as [x, y, z].
        previous_tcp: Previous TCP position as [x, y, z].
        dt: Time interval since last update.
        kAx, kAy: Gains for the admittance control.
        F_Bx, F_By: Measured forces along x and y (replace with sensor readings if available).
        
    Returns:
        A list of angular corrections [delta_roll, delta_pitch, delta_yaw]. (Here, yaw is zero.)
    """
    # Estimate TCP velocity (simple finite difference)
    vB = [(c - p) / dt for c, p in zip(current_tcp, previous_tcp)]
    
    # --- Admittance Component ---
    # Force in y causes rotation about X-axis, force in x causes rotation about Y-axis.
    omega_Ax = kAx * F_By
    omega_Ay = kAy * F_Bx

    # --- Feedforward Component ---
    # For rotation about the X-axis (YZ-plane)
    angle_current_x = math.atan2(current_tcp[1] - fulcrum[1], current_tcp[2] - fulcrum[2])
    angle_future_x  = math.atan2((current_tcp[1] + vB[1]*dt) - fulcrum[1],
                                 (current_tcp[2] + vB[2]*dt) - fulcrum[2])
    omega_ff_x = (angle_future_x - angle_current_x) / dt

    # For rotation about the Y-axis (XZ-plane)
    angle_current_y = math.atan2(current_tcp[0] - fulcrum[0], current_tcp[2] - fulcrum[2])
    angle_future_y  = math.atan2((current_tcp[0] + vB[0]*dt) - fulcrum[0],
                                 (current_tcp[2] + vB[2]*dt) - fulcrum[2])
    omega_ff_y = (angle_future_y - angle_current_y) / dt

    # Total corrections (no yaw correction applied in this example)
    delta_roll = omega_Ax + omega_ff_x    # rotation about X-axis
    delta_pitch = omega_Ay + omega_ff_y     # rotation about Y-axis
    delta_yaw = 0

    return [delta_roll, delta_pitch, delta_yaw]

def compute_pose_and_orientation(
    target_pose: List[float],
    command: int,
    tcp_corr: List[float]
) -> Tuple[float, float, float, List[float]]:
    """
    Computes the pose and orientation based on the target pose and command,
    then adds the TCP correction to the base rotation.
    
    Args:
        target_pose: Target position adjustments [x, y, ...].
        command: Command indicating the movement type.
        tcp_corr: Additional correction [delta_roll, delta_pitch, delta_yaw] from TCP control.
    
    Returns:
        x, y, z and tcp_rotation_rpy list (with the correction applied).
    """
    x, y, z = 0, 0, 0
    rotation_factor = 10

    if command == 1:
        x_rot = math.radians(target_pose[0] * rotation_factor)
        tcp_rotation_rpy = [0, x_rot, 0]
    elif command == 2:
        x_rot = math.radians(target_pose[1] * rotation_factor)
        tcp_rotation_rpy = [0, x_rot, 0]
    elif command == 3:
        y_rot = math.radians(target_pose[2] * rotation_factor)
        tcp_rotation_rpy = [y_rot, 0, 0]
    elif command == 4:
        y_rot = math.radians(target_pose[3] * rotation_factor)
        tcp_rotation_rpy = [y_rot, 0, 0]
    else:
        tcp_rotation_rpy = [0, 0, 0]
    
    # Add TCP pivot correction (admittance + feedforward) to the computed orientation
    tcp_rotation_rpy = [base + corr for base, corr in zip(tcp_rotation_rpy, tcp_corr)]
    
    return x, y, z, tcp_rotation_rpy

def extract_last_tuple(s: str) -> Tuple[float, float]:
    tuples = re.findall(r'\(-?\d+\.?\d*,-?\d+\.?\d*\)', s)
    if tuples:
        last_tuple_str = tuples[-1]
        last_tuple = last_tuple_str.strip('()').split(',')
        return tuple(map(float, last_tuple))
    return None

def apply_target_pose(
    robot: object,
    target_pose: List[float],
    accumulated_pose: List[float],
    origin: m3d.Transform,
    command: int,
    previous_command: int
) -> Tuple[List[float], int]:
    """
    Applies the target pose by computing the required transformation, including the TCP
    pivot correction, and sending it to the robot.
    """
    global prev_tcp, prev_time

    if previous_command != command:
        print('Previous command:', previous_command)
        print('New command:', command)
        origin = set_lookorigin()
        print("Set new origin")
        target_pose[:] = [0] * len(target_pose)
        print("Reset target pose")
    
    accumulated_pose, target_pose = check_boundary(accumulated_pose, target_pose)
    print("Target Pose:", target_pose)
    print("Accumulated Pose:", accumulated_pose)

    # --- Compute Base Orientation without correction ---
    base_x, base_y, base_z, tcp_rotation_rpy_base = compute_pose_and_orientation(target_pose, command, [0, 0, 0])
    
    # --- Compute TCP Pivot Correction ---
    # Get current TCP pose (first three coordinates)
    current_tcp = robot.get_actual_tcp_pose()[:3]
    # Use the origin as the fulcrum; convert its transform to coordinates
    fulcrum = extract_coordinates_from_orientation(origin)[:3]
    
    # Estimate time delta
    current_time = time.time()
    dt = current_time - prev_time if (prev_time is not None and current_time - prev_time > 0) else 0.05

    # Replace these with actual force sensor readings when available
    F_Bx, F_By = 0, 0

    # Compute correction (using example gains 0.8 for both axes)
    tcp_corr = compute_tcp_pivot_correction(current_tcp, fulcrum, prev_tcp, dt, 0.8, 0.8, F_Bx, F_By)
    
    # --- Re-compute Orientation with TCP Correction ---
    _, _, _, tcp_rotation_rpy = compute_pose_and_orientation(target_pose, command, tcp_corr)
    
    # Compute the final transformation using the corrected orientation.
    x, y, z = 0, 0, 0  # These can be adjusted if needed.
    xyz_coords = m3d.Vector(x, y, z)
    tcp_orient = m3d.Orientation.new_euler(tcp_rotation_rpy, encoding='xyz')
    position_vec_coords = m3d.Transform(tcp_orient, xyz_coords)
    oriented_xyz = origin * position_vec_coords 
    coordinates = extract_coordinates_from_orientation(oriented_xyz)
    robot.set_realtime_pose(coordinates)
    
    # Update globals for next iteration
    prev_tcp = current_tcp
    prev_time = current_time

    return accumulated_pose, command

def start_hand_tracking():
    """
    Main loop for receiving server input and moving the robot based on hand tracking.
    """
    global origin, previous_command, prev_tcp, prev_time
    previous_command = 0
    prev_ampli = 0
    accumulated_pose = [0, 0]
    robot_position = [0, 0, 0, 0]
    origin = set_lookorigin()
    
    # Initialize previous TCP and time for velocity estimation
    prev_tcp = robot.get_actual_tcp_pose()[:3]  # Expecting [x, y, z, ...]
    prev_time = time.time()

    robot.init_realtime_control()
    time.sleep(1)

    try:
        while not emergency_stop:
            server_data = get_from_server()
            tup = extract_last_tuple(server_data)
            if tup is None:
                print("Invalid or no input received.")
                continue
            command, position = tup
            command = int(command)
            position = int(position)
            robot_position, prev_ampli = compute_target_pose(robot_position, position, command, prev_ampli)
            accumulated_pose, previous_command = apply_target_pose(robot, robot_position,
                                                                   accumulated_pose, origin,
                                                                   command, previous_command)
    except KeyboardInterrupt:
        print("Stopping hand tracking...")
    finally:
        robot.close()
        print("Robot connection closed.")

def end():
    robot.close()
    print("Robot connection closed")
    client_socket.close()
    print("Client connection closed")
    print("Program ended")

def main():
    robot_set_up()
    home()
    # set_new_tcp(offset=0.275)
    # server_connection()
    # start_hand_tracking()
    end()

if __name__ == '__main__':
    main()

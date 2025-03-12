"""
Face_tracking01
Python program for realtime face tracking of a Universal Robot (tested with UR5cb)
Demonstration Video: https://youtu.be/HHb-5dZoPFQ
Explanation Video: https://www.youtube.com/watch?v=9XCNE0BmtUg

Created by Robin Godwyll
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
with open("robot_variables.yml", "r") as file:
    config = yaml.safe_load(file)

ROBOT_IP = config["connection"]["ip_address"]
ACCELERATION = config["acceleration"]["joint_acceleration"]  # Robot acceleration value
VELOCITY = config["speed"]["joint_speed"]  # Robot speed value

# The Joint position the robot starts at
robot_startposition = [round(math.radians(degree), 3) for degree in config["initial_position"]["joint_angles"]]



# Size of the robot view-window
max_x = 0.1
max_y = 0.2
max_z = 0.2

# Maximum Rotation of the robot at the edge of the view window
hor_rot_max = math.radians(45)
vert_rot_max = math.radians(45)

"""FUNCTIONS _____________________________________________________________________________"""

def set_lookorigin() -> m3d.Transform:
    """
    Creates a new coordinate system at the current robot tcp position.
    This coordinate system is the basis of the face following.
    It describes the midpoint of the plane in which the robot follows faces.

    Return Value:
        orig: math3D Transform Object
            characterises location and rotation of the new coordinate system in reference to the base coordinate system
    """
    position = robot.get_actual_tcp_pose()
    orig = m3d.Transform(position)
    return orig

def extract_coordinates_from_orientation(oriented_xyz: m3d.Transform) -> List[float]:
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
    # print("Received from server: ", data)
    return data

def check_max_xy(xy_coord: List[float]) -> List[float]:
    """
    Checks if the face is outside of the predefined maximum values on the lookaraound plane

    Inputs:
        xy_coord: list of 2 values: x and y value of the face in the lookaround plane.
            These values will be evaluated against max_x and max_y

    Return Value:
        x_y: new x and y values
            if the values were within the maximum values (max_x and max_y) these are the same as the input.
            if one or both of the input values were over the maximum, the maximum will be returned instead
    """
    x_y = [0, 0]

    if -max_x <= xy_coord[0] <= max_x:
        x_y[0] = xy_coord[0]
    elif -max_x > xy_coord[0]:
        x_y[0] = -max_x
    elif max_x < xy_coord[0]:
        x_y[0] = max_x
    else:
        raise Exception(" x is wrong somehow:", xy_coord[0], -max_x, max_x)

    if -max_y <= xy_coord[1] <= max_y:
        x_y[1] = xy_coord[1]
    elif -max_y > xy_coord[1]:
        x_y[1] = -max_y
    elif max_y < xy_coord[1]:
        x_y[1] = max_y
    else:
        raise Exception(" y is wrong somehow", xy_coord[1], max_y)

    return x_y


"""Main Program ____________________________________________________________________________"""
def main():
    robot_set_up()
    home()
    # set_new_tcp(offset= 0.275)
    # server_connection()
    # start_hand_tracking()
    end()

"""FACE TRACKING LOOP ____________________________________________________________________"""

def robot_set_up():
    global robot, robotModel
    print("initialising robot")
    robotModel = URBasic.robotModel.RobotModel()
    robot = URBasic.urScriptExt.UrScriptExt(host=ROBOT_IP, robotModel=robotModel)
    print("robotModel initialised")
    robot.reset_error()
    print("robot initialised")
    time.sleep(1)

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

    Args:
        prev_pose (list): Previous position of the robot [x, y].
        position (float): Position received from the server to adjust movement.
        scale_factor (float): Scaling factor for converting input to real-world measurements.

    Returns:
        list: Target position [x, y].
    """
    target_pose = prev_pose[:]
    position = int(position) 
    if command == 1:
        position = -10
    elif command == 2:
        position = 10
    elif command == 3:
        position = 10
    elif command == 4:
        position = -10
    else:
        position = 0
    
    scale_factor = 0.001

    target_pose[0] = int(position) * scale_factor
    if detect_sign_change(prev_ampli, position):
        print('prev_ampli', prev_ampli)
        print('Current ampli', position)
        target_pose[0] = 0
    prev_ampli = position
    target_pose = check_max_xy(target_pose)
    return target_pose, prev_ampli

# SECTION: Apply Target Pose
def apply_target_pose(
    robot: object,
    target_pose: List[float],
    origin: m3d.Transform,
    command: int,
    previous_command: int
) -> int:
    """
    Applies the target pose by computing the required transformation and sending it to the robot.

    Args:
        robot (object): Robot object for real-time control.
        target_pose (list): Target position [x, y].
        origin (Transform): Reference origin for transformations.
        command (int): Command to determine the pose and orientation.
        previous_command (int): Previous command to check for changes.
    
    Returns:
        int: Updated previous command.
    """
    if previous_command != command:
        print('previous_command', previous_command)
        print('command', command)
        origin = set_lookorigin()
        print("Set new origin")
        target_pose[0] = 0

    x, y, z, tcp_rotation_rpy = compute_pose_and_orientation(target_pose, command)
    previous_command = command
    origin = set_lookorigin()
    
    xyz_coords = m3d.Vector(x, y, z)
    tcp_orient = m3d.Orientation.new_euler(tcp_rotation_rpy, encoding='xyz')
    position_vec_coords = m3d.Transform(tcp_orient, xyz_coords)

    oriented_xyz = origin * position_vec_coords 
    coordinates = extract_coordinates_from_orientation(oriented_xyz)

    robot.set_realtime_pose(coordinates)
    return previous_command

def compute_pose_and_orientation(
    target_pose: List[float],
    command: int
) -> Tuple[float, float, float, List[float]]:
    """
    Computes the pose and orientation based on the command.

    Args:
        target_pose (list): Target position [x, y].
        command (int): Command to determine the pose and orientation.

    Returns:
        tuple: x, y, z coordinates and tcp_rotation_rpy list.
    """
    x, y, z = 0, 0, 0
    rotation_factor = 10

    if command == 1:
        # z = target_pose[0]
        # tcp_rotation_rpy = [0, 0, 0]
        x_rot = math.radians(target_pose[0] * rotation_factor)
        # tcp_rotation_rpy = [0, x_rot, 0]
        tcp_rotation_rpy = [x_rot, 0, 0]
    elif command == 2:
        x_rot = math.radians(target_pose[0] * rotation_factor)
        # tcp_rotation_rpy = [0, x_rot, 0]
        tcp_rotation_rpy = [x_rot, 0, 0]
    elif command == 3:
        x_rot = math.radians(target_pose[0] * rotation_factor)
        # tcp_rotation_rpy = [x_rot, 0, 0]
        tcp_rotation_rpy = [0, x_rot, 0]

    elif command == 4:
        x_rot = math.radians(target_pose[0] * rotation_factor)
        # tcp_rotation_rpy = [x_rot, 0, 0]
        tcp_rotation_rpy = [0, x_rot, 0]
    else:
        tcp_rotation_rpy = [0, 0, 0]
    
    return x, y, z, tcp_rotation_rpy

def extract_last_tuple(s: str) -> Tuple[float, float]:
    tuples = re.findall(r'\(-?\d+\.?\d*,-?\d+\.?\d*\)', s)
    if tuples:
        last_tuple_str = tuples[-1]
        last_tuple = last_tuple_str.strip('()').split(',')
        return tuple(map(float, last_tuple))
    return None

def start_hand_tracking():
    """
    Main loop for receiving server input and moving the robot based on hand tracking.
    """
    global origin, previous_command
    previous_command = 0
    prev_ampli = 0
    robot_position = [0, 0]
    origin = set_lookorigin()

    robot.init_realtime_control()
    time.sleep(1)

    count_amplitude = 0

    try:
        while True:
            command, position = extract_last_tuple(get_from_server())
            position = int(position)
            command = int(command)
            # and keyboard.is_pressed("ctrl")
            if isinstance(position, int):
                robot_position, prev_ampli = compute_target_pose(robot_position, position, command, prev_ampli)
                previous_command = apply_target_pose(robot, robot_position, origin, command, previous_command)
                count_amplitude += prev_ampli
                print("Aplitude:", count_amplitude)
            else:
                print("Invalid or no input received.")
    except KeyboardInterrupt:
        print("Stopping hand tracking...")
    finally:
        robot.close()
        print("Robot connection closed.")

def end():
    robot.close()
    print("Robot Connection Closed")
    client_socket.close()
    print("Client Connection Closed")
    print("Program Ended")

if __name__ == '__main__':
    main()

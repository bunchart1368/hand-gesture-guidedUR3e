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
import threading

"""SERVER FUNCTIONS ______________________________________________________________________"""

def setup_server():
    HOST = 'localhost'
    PORT = 65432
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen()
    return server, HOST, PORT

def handle_client(conn, addr, client_id, start_event, print_lock, client_data, data_lock):
    with conn:
        with print_lock:
            print(f"[CONNECTED] {addr} as Client {client_id}")

        # Wait until start_event is set (i.e. 2 clients are connected)
        start_event.wait()

        # After start_event is released, flush the initial data
        with data_lock:
            if client_id in client_data:
                client_data[client_id] = []

        while True:
            try:
                data = conn.recv(1024).decode().strip()
                if not data:
                    with print_lock:
                        print(f"[DISCONNECTED] Client {client_id} at {addr} (Graceful)")
                    break

                with data_lock:
                    if client_id not in client_data:
                        client_data[client_id] = []
                    client_data[client_id].append(data)

                with print_lock:
                    process_client_data(data)
                    # print(f"[Client {client_id} | {addr}] Received: {data}")

            except (ConnectionResetError, ConnectionAbortedError):
                with print_lock:
                    print(f"[DISCONNECTED] Client {client_id} at {addr} (Error)")
                break

def process_client_data(data: str):
    """
    Process data received from the client.
    """
    print(f"Received data: {data}")
    global emergency_stop


    try:
        command, position = extract_last_tuple(data)
        print(f"Processing Command: {command}, Position: {position}")

        if emergency_stop:
            print("[EMERGENCY STOP] Robot is stopped.")
            return

        # and keyboard.is_pressed("ctrl")
        if isinstance(position, int) and keyboard.is_pressed("ctrl"):
            robot_force_vectors = get_force_sensor_data()
            print("Force Sensor Data: ", robot_force_vectors)
            total_force = total_force_vector(robot_force_vectors)
            print("Total Force: ", total_force)
            if total_force > 30: 
                print("Stop! Force exceeded limit.")
                emergency_stop = True
                return
            robot_position, prev_ampli = compute_target_pose(robot_position, position, command, prev_ampli)
            accumulated_pose, previous_command = apply_target_pose(robot, robot_position, accumulated_pose, origin, command, previous_command)
        else:
            print('Not pressing ctrl')

    except ValueError:
        print("[ERROR] Invalid data format received from client.")

def start_server():
    """
    Start the server to handle client connections.
    """
    server, HOST, PORT = setup_server()

    print_lock = threading.Lock()
    client_lock = threading.Lock()
    data_lock = threading.Lock()
    client_data = {}
    client_threads = []
    client_count = 0
    required_clients = 1
    start_event = threading.Event()  # Used to release clients simultaneously

    global origin, previous_command, prev_ampli, accumulated_pose, robot_position
    previous_command = 0
    prev_ampli = 0
    accumulated_pose = [0, 0]
    robot_position = [0, 0, 0, 0]
    # origin = set_lookorigin()

    # robot.init_realtime_control()
    time.sleep(1)

    print(f"[STARTING] Server is listening on {HOST}:{PORT}")
    server.settimeout(0.1)

    try:
        while True:
            try:
                conn, addr = server.accept()

                with client_lock:
                    client_count += 1
                    client_id = client_count

                thread = threading.Thread(
                    target=handle_client,
                    args=(conn, addr, client_id, start_event, print_lock, client_data, data_lock),
                    daemon=True
                )
                thread.start()
                client_threads.append(thread)

                if client_count == required_clients:
                    print(f"[READY] {required_clients} clients connected. Starting message processing...")
                    start_event.set()  # Signal all clients to begin receiving messages

            except socket.timeout:
                continue

    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Server stopped by user.")
        server.close()

        # Print all client data received
        print("\n[CLIENT DATA DUMP]")
        with data_lock:
            for cid, messages in client_data.items():
                print(f"Client {cid}: {messages}")
    finally:
        robot.close()
        print("Robot connection closed.")




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

# Emergercy stop
emergency_stop = False

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

def get_force_sensor_data():
    robot_force_vector = robot.get_tcp_force()
    # print("Force Sensor Data: ", robot_force_vector)
    return robot_force_vector

def total_force_vector(force_vector) -> float:
    """Compute the total force vector from the force sensor data."""
    fx, fy, fz = force_vector[:3]
    total_force = math.sqrt(fx**2 + fy**2 + fz**2)
    rounted_total_force = round(total_force, 3)
    return rounted_total_force

def euclidean_distance(v1, v2):
    """Compute the Euclidean distance between two vectors."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))

def free_drive_mode():
    robot.freedrive_mode()
    print("Freedrive mode activated")

def end_free_drive_mode():
    robot.end_freedrive_mode()
    print("Freedrive mode deactivated")

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

def check_boundary(accumulated_pose: List[float], target_pose: List[float]):
    # temp_accumulated_pose = [sum(pair) + accumulated_pose[i] for i, pair in enumerate(zip(target_pose[::2], target_pose[1::2]))]
    # for i, pose in enumerate(temp_accumulated_pose):
    #     if i == 0 and (pose > max_right or pose < max_left):
    #         target_pose[:] = [0] * len(target_pose)
    #         # emergency_stop = True
    #         print("reach max right or max left")
    #     if i == 1 and (pose > max_up or pose < max_down):
    #         target_pose[:] = [0] * len(target_pose)
    #         # emergency_stop = True
    #         print("reach max up or max down")
    accumulated_pose = [sum(pair) + accumulated_pose[i] for i, pair in enumerate(zip(target_pose[::2], target_pose[1::2]))]
    return accumulated_pose, target_pose


def check_boundary_2(origin: m3d.Transform, first_position: List[float]):
    max_x_offset = config["end_effector_limits"]["max_x_offset"]
    max_y_offset = config["end_effector_limits"]["max_y_offset"]
    max_z_offset = config["end_effector_limits"]["max_z_offset"]
    max_right = first_position[0] + max_x_offset
    max_left = first_position[0] - max_x_offset
    max_forward = first_position[1] + max_y_offset
    max_backward = first_position[1] - max_y_offset
    max_down = first_position[2] - max_z_offset
    current_x = origin.translation[0]
    current_y = origin.translation[1]
    current_z = origin.translation[2]
    if current_x > max_right or current_x < max_left:
        print("X out of bounds")
    if current_y > max_forward or current_y < max_backward:
        print("Y out of bounds")
    if current_z > max_down:
        print("Z out of bounds")

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

# SECTION: Apply Target Pose
def apply_target_pose(
    robot: object,
    target_pose: List[float],
    accumulated_pose: List[float],
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
        target_pose[:] = [0] * len(target_pose)
        print("Reset Target Pose")
    
    accumulated_pose, target_pose = check_boundary(accumulated_pose, target_pose)
    # print("Target Pose: ", target_pose)
    # print("Accumulated Pose: ", accumulated_pose)

    x, y, z, tcp_rotation_rpy = compute_pose_and_orientation(target_pose, command)
    previous_command = command
    origin = set_lookorigin()
    # check_boundary_2(origin, first_position)
    
    xyz_coords = m3d.Vector(x, y, z)
    tcp_orient = m3d.Orientation.new_euler(tcp_rotation_rpy, encoding='xyz')
    position_vec_coords = m3d.Transform(tcp_orient, xyz_coords)

    oriented_xyz = origin * position_vec_coords 
    coordinates = extract_coordinates_from_orientation(oriented_xyz)

    robot.set_realtime_pose(coordinates)
    return accumulated_pose, previous_command

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
    accumulated_pose = [0, 0]
    robot_position = [0, 0, 0, 0]
    origin = set_lookorigin()

    robot.init_realtime_control()
    time.sleep(1)

    try:
        while not(emergency_stop):
            command, position = extract_last_tuple(get_from_server())
            position = int(position)
            command = int(command)
            # and keyboard.is_pressed("ctrl")
            if isinstance(position, int) and keyboard.is_pressed("ctrl"):
                robot_force_vectors = get_force_sensor_data()
                print("Force Sensor Data: ", robot_force_vectors)
                total_force = total_force_vector(robot_force_vectors)
                print("Total Force: ", total_force)
                if total_force > 30: 
                    print("Stop! Force exceeded limit.")
                    emergency_stop = True
                    break
                robot_position, prev_ampli = compute_target_pose(robot_position, position, command, prev_ampli)
                accumulated_pose, previous_command = apply_target_pose(robot, robot_position, accumulated_pose, origin, command, previous_command)
            else:
                None

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

def set_up_test_environment():
    position1 = robot.get_actual_tcp_pose()
    print("Current TCP Position: ", position1[:3])
    free_drive_mode()
    # keyboard.wait('q')  # Blocks until 'q' is pressed
    time.sleep(10)  # Simulate free drive mode for 10 seconds
    print("10 seconds in free drive mode")
    end_free_drive_mode()
    position2 = robot.get_actual_tcp_pose()
    print("Current TCP Position: ", position2[:3])
    distance = euclidean_distance(position1[:3], position2[:3])
    print("Distance between positions: ", distance)
    tcp_offset = config["end_effector"]["offset"] - distance
    set_new_tcp(tcp_offset)
    return position1, position2, tcp_offset

def read_force_sensor_loop():
    while True:
        robot_force_vectors = get_force_sensor_data()
        # print("Force Sensor Data: ", robot_force_vectors)
        total_force = total_force_vector(robot_force_vectors)
        print("Total Force: ", total_force)
        time.sleep(0.1)

def main():
    # robot_set_up()
    # home()
    # set_new_tcp(offset= config["end_effector"]["offset"])
    # set_up_test_environment()
    # server_connection()
    # start_hand_tracking()
    start_server()
    end()

if __name__ == '__main__':
    main()

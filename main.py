"""
Face_tracking01
Python program for realtime face tracking of a Universal Robot (tested with UR5cb)
Demonstration Video: https://youtu.be/HHb-5dZoPFQ
Explanation Video: https://www.youtube.com/watch?v=9XCNE0BmtUg

Created by Robin Godwyll
License: GPL v3 https://www.gnu.org/licenses/gpl-3.0.en.html

"""

import URBasic
import math
import numpy as np
import sys
import cv2
import time
import imutils
from imutils.video import VideoStream
import math3d as m3d
import re, socket

"""SETTINGS AND VARIABLES ________________________________________________________________"""

RASPBERRY_BOOL = False
# If this is run on a linux system, a picamera will be used.
# If you are using a linux system, with a webcam instead of a raspberry pi delete the following if-statement
if sys.platform == "linux":
    import picamera
    from picamera.array import PiRGBArray
    RASPBERRY_BOOL = True

# ROBOT_IP = '192.168.1.112'
ROBOT_IP = '10.10.0.61'
ACCELERATION = 0.9  # Robot acceleration value
VELOCITY = 0.8  # Robot speed value

# The Joint position the robot starts at
# robot_startposition = (math.radians(-218),
#                     math.radians(-63),
#                     math.radians(-93),
#                     math.radians(-20),
#                     math.radians(88),
#                     math.radians(0))
robot_startposition = [round(math.radians(degree), 3) for degree in [90, -90, -90, -45, 90, 0]]

# Variable which scales the robot movement from pixels to meters.
# m_per_pixel = 00.00009  
m_per_pixel = 00.000009 #Add more 0  


# Size of the robot view-window
# The robot will at most move this distance in each direction
max_x = 0.2
max_y = 0.2
max_z = 0.2

# Maximum Rotation of the robot at the edge of the view window
hor_rot_max = math.radians(90)
vert_rot_max = math.radians(90)
z_rot_max = math.radians(90)


"""FUNCTIONS _____________________________________________________________________________"""

def check_max_xy(xy_coord):
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
    #print("xy before conversion: ", xy_coord)

    if -max_x <= xy_coord[0] <= max_x:
        # checks if the resulting position would be outside of max_x
        x_y[0] = xy_coord[0]
    elif -max_x > xy_coord[0]:
        x_y[0] = -max_x
    elif max_x < xy_coord[0]:
        x_y[0] = max_x
    else:
        raise Exception(" x is wrong somehow:", xy_coord[0], -max_x, max_x)

    if -max_y <= xy_coord[1] <= max_y:
        # checks if the resulting position would be outside of max_y
        x_y[1] = xy_coord[1]
    elif -max_y > xy_coord[1]:
        x_y[1] = -max_y
    elif max_y < xy_coord[1]:
        x_y[1] = max_y
    else:
        raise Exception(" y is wrong somehow", xy_coord[1], max_y)
    #print("xy after conversion: ", x_y)

    return x_y

def set_lookorigin():
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

def extract_coordinates_from_orientation(oriented_xyz):
    oriented_xyz_coord = oriented_xyz.pose_vector

    coordinates_str = str(oriented_xyz_coord)
    numbers = re.findall(r"-?\d+\.\d+", coordinates_str)

    # Convert extracted strings to float
    coordinates = [float(num) for num in numbers]
    return coordinates

def server_connection():
    global client_socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('localhost', 12345))
    print("Connected to the server.")
    
def get_from_server():
    data = client_socket.recv(1024).decode()
    print("Received from server: ", data)
    return data

"""FACE TRACKING LOOP ____________________________________________________________________"""

def robot_set_up():
    global robot, robotModel
    # initialise robot with URBasic
    print("initialising robot")
    robotModel = URBasic.robotModel.RobotModel()
    robot = URBasic.urScriptExt.UrScriptExt(host=ROBOT_IP,robotModel=robotModel)

    robot.reset_error()
    print("robot initialised")
    time.sleep(1)

def home():
    robot.movej(q=robot_startposition, a= ACCELERATION, v= VELOCITY )
def compute_target_pose(prev_pose, position, scale_factor=m_per_pixel):
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
    # Convert server input to meters and add to previous pose
    position = int(position)
    position = -300 + (250 - (-300)) * (position - 10) / (70 - 10)
    target_pose[0] += int(position) * scale_factor  # Modify x based on input
    # Clamp to ensure within max bounds
    target_pose = check_max_xy(target_pose)
    return target_pose


def apply_target_pose(robot, target_pose, origin, command):
    """
    Applies the target pose by computing the required transformation and sending it to the robot.

    Args:
        robot (object): Robot object for real-time control.
        target_pose (list): Target position [x, y].
        origin (Transform): Reference origin for transformations.
    """
    if command == 1:
        x = 0
        y = 0
        z = target_pose[0]  # Assuming flat movement plane
        # Compute percentage-based rotation limits
        x_rot = (x / max_x) * hor_rot_max
        y_rot = (y / max_y) * vert_rot_max * -1
        # Create orientation
        tcp_rotation_rpy = [0, 0, 0]
    elif command == 2:
        x = 0
        y = 0
        z = 0  # Assuming flat movement plane
        # Compute percentage-based rotation limits
        x_rot = (target_pose[0] / max_x) * hor_rot_max
        y_rot = (y / max_y) * vert_rot_max * -1
        # Create orientation
        tcp_rotation_rpy = [0, 0, x_rot]
    elif command == 3:
        x = 0
        y = 0
        z = 0  # Assuming flat movement plane
        # Compute percentage-based rotation limits
        # x_rot = (target_pose[0] / max_x) * hor_rot_max
        x_rot = (target_pose[0] / max_x) * hor_rot_max* 0.4
        y_rot = (y / max_y) * vert_rot_max * -1
        # Create orientation
        tcp_rotation_rpy = [x_rot, 0, 0]
    elif command == 4:
        x = 0
        y = 0
        z = target_pose[0]  # Assuming flat movement plane
        # Compute percentage-based rotation limits
        # x_rot = (x / max_x) * hor_rot_max 
        x_rot = (x / max_x) * hor_rot_max * 0.4
        y_rot = (y / max_y) * vert_rot_max * -1
        # Create orientation
        tcp_rotation_rpy = [0, x_rot, 0]
    
    # Create vector for position
    xyz_coords = m3d.Vector(x, y, z)
    tcp_orient = m3d.Orientation.new_euler(tcp_rotation_rpy, encoding='xyz')
    position_vec_coords = m3d.Transform(tcp_orient, xyz_coords)

    # Transform based on origin
    oriented_xyz = origin * position_vec_coords
    coordinates = extract_coordinates_from_orientation(oriented_xyz)

    # Send target pose to robot
    robot.set_realtime_pose(coordinates)

def extract_last_tuple(s):
    # Find all tuples in the string, including those with floating-point numbers
    tuples = re.findall(r'\(-?\d+\.?\d*,-?\d+\.?\d*\)', s)
    # Return the last tuple if any are found
    if tuples:
        last_tuple_str = tuples[-1]
        # Remove parentheses and split by ','
        last_tuple = last_tuple_str.strip('()').split(',')
        # Convert to floats and return as a tuple
        return tuple(map(float, last_tuple))
    return None
def start_hand_tracking():
    """
    Main loop for receiving server input and moving the robot based on hand tracking.
    """
    global origin
    robot_position = [0, 0]  # Initialize position in 2D plane
    origin = set_lookorigin()  # Set the origin

    robot.init_realtime_control()
    time.sleep(1)  # Allow time for initialization

    try:
        print("Starting hand tracking loop...")
        while True:
            command,position = extract_last_tuple(get_from_server())
            position = int(position)
            command = int(command)
            print(f"Received position: {position}")
            if isinstance(position, int):  # Ensure valid numeric input
                print(f"Received position: {position}")
                robot_position = compute_target_pose(robot_position, position)
                apply_target_pose(robot, robot_position, origin, command)
                print("Robot moved to target position.")
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
    robot_set_up()
    home()
    server_connection()
    start_hand_tracking()
    end()

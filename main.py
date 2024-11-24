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

ROBOT_IP = '192.168.1.112'
# ROBOT_IP = '10.10.0.61'
ACCELERATION = 0.9  # Robot acceleration value
VELOCITY = 0.8  # Robot speed value

# The Joint position the robot starts at
# robot_startposition = (math.radians(-218),
#                     math.radians(-63),
#                     math.radians(-93),
#                     math.radians(-20),
#                     math.radians(88),
#                     math.radians(0))
robot_startposition = [round(math.radians(degree), 3) for degree in [90, -90, -90, -45, 90, 45]]
# Path to the face-detection model:
pretrained_model = cv2.dnn.readNetFromCaffe("MODELS/deploy.prototxt.txt", "MODELS/res10_300x300_ssd_iter_140000.caffemodel")

video_resolution = (700, 400)  # resolution the video capture will be resized to, smaller sizes can speed up detection
video_midpoint = (int(video_resolution[0]/2),
                  int(video_resolution[1]/2))
video_asp_ratio  = video_resolution[0] / video_resolution[1]  # Aspect ration of each frame
video_viewangle_hor = math.radians(25)  # Camera FOV (field of fiew) angle in radians in horizontal direction

# Variable which scales the robot movement from pixels to meters.
m_per_pixel = 00.00009  

# Size of the robot view-window
# The robot will at most move this distance in each direction
max_x = 0.2
max_y = 0.2

# Maximum Rotation of the robot at the edge of the view window
hor_rot_max = math.radians(50)
vert_rot_max = math.radians(25)


"""FUNCTIONS _____________________________________________________________________________"""

def server_connection():
    global client_socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('localhost', 12345))
    print("Connected to the server.")

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

def move_to_face(list_of_facepos,robot_pos):
    """
    Function that moves the robot to the position of the face

    Inputs:
        list_of_facepos: a list of face positions captured by the camera, only the first face will be used
        robot_pos: position of the robot in 2D - coordinates

    Return Value:
        prev_robot_pos: 2D robot position the robot will move to. The basis for the next call to this funtion as robot_pos
    """


    face_from_center = list(list_of_facepos[0])  # TODO: find way of making the selected face persistent

    prev_robot_pos = robot_pos
    scaled_face_pos = [c * m_per_pixel for c in face_from_center]

    robot_target_xy = [a + b for a, b in zip(prev_robot_pos, scaled_face_pos)]
    # print("..", robot_target_xy)

    robot_target_xy = check_max_xy(robot_target_xy)
    prev_robot_pos = robot_target_xy
    print("Robot Target: ", robot_target_xy)

    # x = robot_target_xy[0]
    # y = robot_target_xy[1]
    z = 0
    x = 0
    y = 0
    # z = robot_target_xy[0] #(-50,50)
    xyz_coords = m3d.Vector(x, y, z)

    x_pos_perc = x / max_x
    y_pos_perc = y / max_y

    x_rot = x_pos_perc * hor_rot_max
    y_rot = y_pos_perc * vert_rot_max * -1

    # tcp_rotation_rpy = [y_rot, x_rot, 0]
    # tcp_rotation_rpy = [0, 0, 0]
    # tcp_rotation_rpy = [0, 0, robot_target_xy[0]]
    tcp_rotation_rpy = [robot_target_xy[0], 0, 0]
    # tcp_rotation_rpy = [0, robot_target_xy[0], 0]

    tcp_orient = m3d.Orientation.new_euler(tcp_rotation_rpy, encoding='xyz')
    position_vec_coords = m3d.Transform(tcp_orient, xyz_coords)
    print("Position: ", position_vec_coords)
    print("Origin: ", origin)

    oriented_xyz = origin * position_vec_coords
    print("Orientation: ", oriented_xyz)
    coordinates = extract_coordinates_from_orientation(oriented_xyz)

    qnear = robot.get_actual_joint_positions()
    next_pose = coordinates
    robot.set_realtime_pose(next_pose)
    # robot.set_realtime_pose([-0.1610,  0.3119,  0.5579,   -1.4006, -0.6485,  0.7004])

    return prev_robot_pos

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

def strat_hand_tracking():
    robot_position = [0,0]
    origin = set_lookorigin()

    robot.init_realtime_control()  # starts the realtime control loop on the Universal-Robot Controller
    time.sleep(1) # just a short wait to make sure everything is initialised

    try:
        print("starting loop")
        while True:

            # ret, frame = cap.read()
            # if not ret:
            #     print("Error: Could not read frame from camera.")
            #     break
            server_connection()
            position = get_from_server()
            print("frame shown")
            if position:
                robot_position = move_to_face(position,robot_position)
            else:
                print("No face found")
            print("robot moved")

        print("exiting loop")
    except KeyboardInterrupt:
        print("closing robot connection")
        # Remember to always close the robot connection, otherwise it is not possible to reconnect
        robot.close()

    except:
        robot.close()

if __name__ == '__main__':
    main()

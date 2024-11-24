"""
Face_tracking01
Python program for controlling a Universal Robot (tested with UR5cb)

Created by Robin Godwyll
License: GPL v3 https://www.gnu.org/licenses/gpl-3.0.en.html

"""
import URBasic
import math
import numpy as np
import sys
import time
import math3d as m3d

"""SETTINGS AND VARIABLES ________________________________________________________________"""

# ROBOT_IP = '192.168.1.106'
ROBOT_IP = '10.10.0.61'
ACCELERATION = 3  # Robot acceleration value
VELOCITY = 9  # Robot speed value

# The Joint position the robot starts at
# robot_startposition = (math.radians(-218),
                    # math.radians(-63),
                    # math.radians(-93),
                    # math.radians(-20),
                    # math.radians(88),
                    # math.radians(0))
robot_startposition = [round(math.radians(degree), 3) for degree in [90, -90, -90, 0, 90, 360]]

"""ROBOT CONTROL LOOP ____________________________________________________________________"""

# initialise robot with URBasic
print("initialising robot")
robotModel = URBasic.robotModel.RobotModel()
robot = URBasic.urScriptExt.UrScriptExt(host=ROBOT_IP, robotModel=robotModel)

robot.reset_error()
print("robot initialised")
time.sleep(1)

# Move Robot to the midpoint of the lookplane
robot.movej(q=robot_startposition, a=ACCELERATION, v=VELOCITY)
print("Start position set", robot_startposition)
robot_position = [0, 0]

# robot.init_realtime_control()  # starts the realtime control loop on the Universal-Robot Controller
# time.sleep(1)  # just a short wait to make sure everything is initialised
# position_test = robot.get_inverse_kin([ 0.0997,  0.4605,  0.4805,   -1.5702, -0.0995,  0.1006])
# print("Position Test:", position_test)
# start = 1
# for i in range(10):
#     if start % 2 == 0:
#         print('Iteration:', i)
#         test_1 = robot.get_inverse_kin([ 0.0997,  0.4605,  0.4805,   -1.5702, -0.0995,  0.1006])
#         robot.movej(q=test_1, a=ACCELERATION, v=VELOCITY)
#     else:
#         test_2 = robot.get_inverse_kin([ 0,  0.4605,  0.4805,   -1.5702, -0.0995,  0.1006])
#         robot.movej(q=test_2, a=ACCELERATION, v=VELOCITY)
#         print('Iteration:', i)
#     start += 1
# position_test =np.array([ -0.01,  0,0,0,0,0])
# position = (np.array([ 0.0997,  0.4605,  0.4805,   -1.5702, -0.0995,  0.1006])+position_test)
# for i in range(10):
#         test_2 = robot.get_inverse_kin(position)
#         print('Test 2:', test_2)
#         robot.movej(q=test_2, a=ACCELERATION, v=VELOCITY)
#         # robot.set_realtime_pose(test_2)
#         position += position_test
#         print('Iteration:', i)
# position_test =np.array([ -0.01,  0,0,0,0,0])
# position = (np.array([1.571, -1.571, -1.571, 0.0, 1.571, 6.283])+position_test).tolist()
# print('Position_datatype:', position.__class__)
# for i in range(10):
#         # test_2 = robot.get_inverse_kin(position)
#         test_2 = position
#         print('Test 2:', test_2)
#         robot.set_realtime_pose(test_2)
#         # robot.set_realtime_pose(test_2)
#         position += position_test
#         print('Iteration:', i)
robot.close()

import URBasic

import time, math
import math3d as m3d
from gripper import Gripper
import numpy as np

# Robot IP address
ROBOT_IP = '10.10.0.61'
# ROBOT_IP = '192.168.1.109'

# Initialize robot with URBasic
print("initialising robot")
robotModel = URBasic.robotModel.RobotModel()
robot = URBasic.urScriptExt.UrScriptExt(host=ROBOT_IP, robotModel=robotModel)

robot.reset_error()
print("robot initialised")
time.sleep(1)

def gripper_connection():
        global gripper
        gripper = Gripper('10.10.0.61', 63352)
        gripper.connection()

def gripper_test():
        gripper.control(255)
        time.sleep(3)
        gripper.control(0)

def gripper_close():
        gripper.control(255)

def gripper_open():
        gripper.control(0)
def pose_add(a,b):
       a = np.around(np.array(a),3)
       b = np.around(np.array(b),3)

       return np.add(a,b)
def round_list(List, len = 4):
       return [round(num, len) for num in List]
       



if __name__ == '__main__':
    # gripper_connection()
    # gripper_open()
    # time.sleep(3)
    # gripper_close()

#     # Move Robot to a safe start position (optional)
#     robot_startposition = (
#         math.radians(-218),
#         math.radians(-63),
#         math.radians(-93),
#         math.radians(-20),
#         math.radians(88),
#         math.radians(90)
#     )

    robot_startposition = [round(math.radians(degree), 3) for degree in [90, -90, -90, 0, 90, 360]] #Home
    robot.movej(q=robot_startposition, a=0.9, v=0.8)

    # Initialize real-time control
    robot.init_realtime_control()
    time.sleep(1)

    pose = round_list(robot.get_actual_tcp_pose())
    print('Current pose: ', pose)
#     offset_pose = [0.1,0.1,0.1,0.1,0.1,0.1]
    offset_pose = [0,0,0.04,0,0,0]
    new_pose = round_list(pose_add(pose, offset_pose))
    print('Current pose2: ', new_pose)
    new_pose_2 = round_list(pose_add(new_pose, offset_pose))
    robot.movej(q = robot.get_inverse_kin(new_pose), a=0.9, v=0.8)
#     robot.set_realtime_pose([1.5728, 0.0019, -0.0008,0.1318, 0.4626, 0.4804])
#     robot.set_realtime_pose([-1.573, 0.002, -0.001,0.132, 0.463, 0.52])
#    robot.movej(q = robot.get_inverse_kin(new_pose), a=9, v=9)
#    robot.movej(q = robot.get_inverse_kin(pose), a=9, v=9)
        # time.sleep(1)
#    time.sleep(1)
        # robot.set_realtime_pose([-1.573, 0.002, -0.001,0.132, 0.463, 0.52])
#    time.sleep(1)
#    robot.set_realtime_pose(new_pose_2)
#    time.sleep(1)
        # print('set_realtime_pose: ', str(new_pose))

    # print('go_position: ', go_position)
    # while True:
        # robot.set_realtime_pose([-0.1709,  0.3119,  0.5580,   -1.7454,  0.6407,  1.8717])
        # robot.set_realtime_pose([-0.1709,  0.3119,  0.4580,   -1.7454,  0.6407,  1.8717])
        # robot.set_realtime_pose([-0.25,  0.3119,  0.3580,   -1.7454,  0.6407,  1.8717])
    # robot.set_realtime_pose(([ 0.1318,  0.4626,  0.4805,   -1.5726,  0.0019, -0.0007]))


    # Wait for a few seconds to observe the robot's movement
    # time.sleep(5)

    # Close the robot connection
    robot.close()
    # exit(1)
    # sys.exit("Program terminated due to robot disconnection.")
    print("robot connection closed")

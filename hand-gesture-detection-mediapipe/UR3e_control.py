#####################################################################################################################

import socket, struct, time
from pymodbus.client import ModbusTcpClient
import numpy,time
import math


def UR_set_up():
        global tcp, joint_rad, joint_deg, joint_rev
        robot = '10.10.0.61'
        port = 30003
        gripper_port    = 63352
        tcp = {}
        joint_rad = {}
        joint_deg = {}
        joint_rev = {}

        stop_robot = False

        #Establish connection to controller
        global s, client
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.connect((robot, port))
        print("Start")

        client = ModbusTcpClient(robot)

        if client.connect() :
                print ('Connection established')
        else :
                print ('Connection failed')
#####################################################################################################################
                
def read_pos():


        #Read tcp value from Modbus server
        t = client.read_holding_registers(400,6)
        
        #Extract and convert values
        for i in range(0,6):
                tcp[i] = t.registers[i]
                if tcp[i] > 32768:
                        tcp[i] = tcp[i] - 65535
                if i < 3 :
                        tcp[i] = float(tcp[i])/10000
                else :
                        tcp[i] = float(tcp[i])/1000
        #Read joint angles from Modbus server
        j = client.read_holding_registers(270,6)                        
        r = client.read_holding_registers(320,6)        
        #Extract and convert values	
        for i in range(0,6):	
                joint_rad[i] = j.registers[i]
                joint_rev[i] = r.registers[i]

                if joint_rev[i] == 65535:
                        joint_rad[i] = joint_rad[i] - 6283             
                joint_rad[i] = float(joint_rad[i])/1000                     
                joint_deg[i] = "{:.2f}".format(numpy.degrees(joint_rad[i]))

        if 1:
                #Print title to screen
                print ('*******')
                print ('UR Controller Primary Client Interface Reader' )
                print ('*******')
                #extract packet length, timestamp and packet type from start of packet and print to screen
                #if message type is cartesian data, extract doubles for 6DOF pos of TCP and print to screen
                print ('X  ' +  '%20s' % str(tcp[0]))
                print ('Y  ' +  '%20s' % str(tcp[1]))
                print ('Z  ' +  '%20s' % str(tcp[2]))
                print ('RX ' +  '%20s' % str(tcp[3]))
                print ('RY ' +  '%20s' % str(tcp[4]))
                print ('RZ ' +  '%20s' % str(tcp[5]))
                print ('****************************************\n')

                print ('Base    ' +  '%20s' % str(joint_deg[0]))
                print ('Shoulder' +  '%20s' % str(joint_deg[1]))
                print ('Elbow   ' +  '%20s' % str(joint_deg[2]))
                print ('Wrist 1'  +  '%20s' % str(joint_deg[3]))
                print ('Wrist 2'  +  '%20s' % str(joint_deg[4]))
                print ('Wrist 3'  +  '%20s' % str(joint_deg[5]))
                print ('****************************************\n')
                time.sleep(1)

def robot_turn(input_int):
        # read_pos()
        if input_int == 0:
                print("Zero command")
                cmd_move = str.encode('x = get_actual_joint_positions()\n')
                s.send(cmd_move)
                cmd_move = str.encode('movej([x[0],x[1],x[2],x[3],x[4],x[5]+0.05]), 0.5, 0.2,0.01, 0.1, 300)\n')
                s.send(cmd_move)
        if input_int == 1:
                print("First command")
                s.send(b'movel(pose_add(get_actual_tcp_pose(),p[0.03,0,0,0,0,0]),0.5,0.2,0,0)\n')
                time.sleep(0.5)
        if input_int == 2:
                print("Second command")
                s.send(b'movel(pose_add(get_actual_tcp_pose(),p[-0.03,0,0,0,0,0]),0.5,0.2,0,0)\n')
                time.sleep(0.5)
def home():
        count = 0
        while count < 5000:
                print("First command")
                radians_list = [round(math.radians(degree), 3) for degree in [-90, -90, -90, -90, 90, 0]]
                cmd_move = str.encode(f'servoj({radians_list}, 0, 0,0.01, 0.1, 200)\n')
                s.send(cmd_move)
                print("Done send command")
                count += 1
        print("Home command")

#####################################################################################################################

if __name__ == '__main__':
        UR_set_up()
        # robot_turn()
        # robot_turn(0)
        home()


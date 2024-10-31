import socket
import time

class Robot:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.r = None
    
    def connection(self):
        #### Establish connection to controller
        self.r = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.r.connect((self.ip, self.port))
        recv_data = self.r.recv(1024)
        if recv_data:
            print('Connected to Robot RTDE....SUCCESSFULLY!')
        else:
            print('Connected to Robot RTDE...FAILED!')
    
    def move(self, x, y, rz, mode=0):
        moveX=.046
        moveY=-.32
        moveZ=.05
        moveRx=2.2
        moveRy=2.24
        moveRz=0

        #Camera offset
        cameraY=(y*0.79-20.54)/1000 #mm
        cameraX=(x*0.8936-90)/1000 #mm

        #Robot offsets
        offsetZ= - 0.250
        offsetX= 0.250

        vel=0.5
        if mode == 0:
               moveZ = offsetZ+0.066
               moveX = offsetX+cameraX
               moveY+=cameraY
        elif mode == 1:
               moveZ = offsetZ
               moveX = offsetX+cameraX
               moveY+=cameraY
        elif mode == 2:
               moveZ = moveZ
               moveX = offsetX+cameraX
               moveY+=cameraY
               vel=0.4

        cmd_move = str.encode('movel(p['+str(moveX)+','+str(moveY)+','+str(moveZ)+','+str(moveRx)+','+str(moveRy)+','+str(moveRz)+'],0.5,{},0,0)\n'.format(vel))
        self.r.send(cmd_move)
        time.sleep(1.6)

    def home(self):
        print('Robot start moving')
        moveX=.046
        moveY=-.32
        moveZ=.05
        moveRx=2.2
        moveRy=2.24
        moveRz=0

        cmd_move = str.encode('movel(p['+str(moveX)+','+str(moveY)+','+str(moveZ)+','+str(moveRx)+','+str(moveRy)+','+str(moveRz)+'],0.5,0.5,0,0)\n')
        #r.send(b'movel(p[0.2,-0.35,0.1,2.253,-2.271,0],0.5,0.25,0,0)\n')
        print (cmd_move)
        self.r.send(cmd_move)
        time.sleep(1)

if __name__ == "__main__":
    robot = Robot()
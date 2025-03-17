import socket
import time

class Gripper:
    def __init__(self, ip, port) -> None:
        self.ip = ip
        self.port = port
        self.g = None

    def connection(self):
        self.g = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.g.connect((self.ip, self.port))
        self.g.sendall(b'GET POS\n')
        g_recv = str(self.g.recv(10), 'UTF-8')
        if g_recv:
            self.g.send(b'SET ACT 1\n')
            g_recv = str(self.g.recv(10), 'UTF-8')
            print(g_recv)
            time.sleep(3)
            self.g.send(b'SET GTO 1\n')
            self.g.send(b'SET SPE 255\n')
            self.g.send(b'SET FOR 255\n')
            print('Gripper Activated')

    def control(self, position):
        if position == 0:
            self.g.send(b'SET POS 0\n')
        elif position == 255:
            self.g.send(b'SET POS 255\n')
        else:
            cmd = str.encode(f'SET POS {position}\n')
            self.g.send(cmd)
        g_recv = str(self.g.recv(10), 'UTF-8')
        self.g.send(b'GET POS \n')
        g_recv = str(self.g.recv(10), 'UTF-8')
        print('Gripper Pos = ' + g_recv)

if __name__ == "__main__":
    gripper = Gripper('10.10.0.61', 63352)
    gripper.connection()
    gripper.control(0)
    time.sleep(6)
    gripper.control(255)


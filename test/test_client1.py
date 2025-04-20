import socket
import random
import time

def send_random_numbers(host='127.0.0.2', port=65432):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        print("[Client 1] Connected to server.")

        while True:
            # number = random.randint(1, 100)
            number = 1.001
            print(f"[Client 1] Sending number: {number}")
            # send_command = f"{number}gg"
            send_command = f"(2, {number});"
            s.sendall(send_command.encode())
            
if __name__ == "__main__":
    send_random_numbers()

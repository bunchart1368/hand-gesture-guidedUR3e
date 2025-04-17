import socket
import random
import time

def send_random_numbers(host='127.0.0.1', port=65432):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        print("[Client 2] Connected to server.")

        while True:
            number = random.randint(100, 200)
            print(f"[Client 2] Sending number: {number}")
            send_command = f"{number}hh"
            s.sendall(send_command.encode())

if __name__ == "__main__":
    send_random_numbers()

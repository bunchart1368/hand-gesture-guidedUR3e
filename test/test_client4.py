import socket
import random
import time

def send_random_numbers(host='localhost', port=12345):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        print("[Client 2] Connected to server.")

        while True:
            number = random.randint(100, 200)
            print(f"[Client 2] Sending number: {number}")
            s.sendall(str(number).encode())

            data = s.recv(1024).decode()
            print(f"[Client 2] Server response: {data}")
            # time.sleep(1.5)

if __name__ == "__main__":
    send_random_numbers()

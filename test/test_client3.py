import socket
import random
import time

def send_random_numbers(host='localhost', port=12345):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        print("Connected to server.")

        while True:
            number = random.randint(1, 100)
            print(f"Sending number: {number}")
            s.sendall(str(number).encode())

            data = s.recv(1024).decode()
            print(f"Server response: {data}")
            # time.sleep(1)  # send one number every second

if __name__ == "__main__":
    send_random_numbers()

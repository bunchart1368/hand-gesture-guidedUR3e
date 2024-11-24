import socket
import time
import random

def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('localhost', 12345))
    server_socket.listen(1)
    print("Server is waiting for a connection...")

    conn, addr = server_socket.accept()
    print(f"Connected to {addr}")

    while True:
        # Generate a random command (replace this with actual hand detection logic)
        command = str(random.randint(-50, 50))
        conn.send(command.encode())
        print(f"Sent Command: {command}")
        time.sleep(1)  # Simulate delay for real-time command sending
    conn.close()
    server_socket.close()

if __name__ == "__main__":
    start_server()

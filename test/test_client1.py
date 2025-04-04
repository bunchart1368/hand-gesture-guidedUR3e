import socket, time, re


def start_client():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('localhost', 12345))
    print("Connected to the server.")

    while True:
        command = client_socket.recv(1024).decode()
        print(f"Received Command: {command}")
        # Perform actions based on the received command
        # time.sleep(3)
    client_socket.close()

if __name__ == "__main__":
    start_client()

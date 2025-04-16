import socket
import threading
import time

def handle_client(conn, addr):
    print(f"[NEW CONNECTION] {addr} connected.")
    with conn:
        while True:
            try:
                data = conn.recv(1024).decode()
                if not data:
                    print(f"[DISCONNECTED] {addr}")
                    break
                print(f"[{addr}] Received: {data}")
                response = f"ACK: {data}"
                conn.sendall(response.encode())
            except (ConnectionResetError, ConnectionAbortedError):
                print(f"[ERROR] Connection lost with {addr}")
                break

def start_server(host='localhost', port=12345):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind((host, port))
        server.listen()
        server.settimeout(0.1)  # Wait max 1 second for accept()

        print(f"[LISTENING] Server is listening on {host}:{port}")

        try:
            while True:
                try:
                    conn, addr = server.accept()
                    thread = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
                    thread.start()
                    print(f"[ACTIVE CONNECTIONS] {threading.active_count() - 1}")
                except socket.timeout:
                    continue  # Timeout allows checking for Ctrl+C
        except KeyboardInterrupt:
            print("\n[SHUTDOWN] Server is shutting down gracefully...")

if __name__ == "__main__":
    start_server()

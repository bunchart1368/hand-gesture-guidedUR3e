import socket
import threading

def setup_server():
    # Define server parameters
    HOST = 'localhost'
    PORT = 65432

    # Create and configure server socket
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen()

    return server, HOST, PORT

def handle_client(conn, addr, client_id, print_lock):
    with conn:
        print(f"[CONNECTED] {addr} as Client {client_id}")
        while True:
            try:
                data = conn.recv(1024).decode().strip()
                if not data:
                    break

                with print_lock:
                    print(f"[Client {client_id} | {addr}] Received: {data}")

            except (ConnectionResetError, ConnectionAbortedError):
                with print_lock:
                    print(f"[DISCONNECTED] Client {client_id} at {addr}")
                break

def start_server():
    # Set up server
    server, HOST, PORT = setup_server()

    print_lock = threading.Lock()
    client_count = 0
    client_lock = threading.Lock()

    print(f"[STARTING] Server is listening on {HOST}:{PORT}")
    server.settimeout(0.1)

    try:
        while True:
            try:
                conn, addr = server.accept()

                with client_lock:
                    client_count += 1
                    client_id = client_count

                thread = threading.Thread(target=handle_client, args=(conn, addr, client_id, print_lock), daemon=True)
                thread.start()

            except socket.timeout:
                continue

    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Server stopped by user.")
        server.close()

if __name__ == "__main__":
    start_server()

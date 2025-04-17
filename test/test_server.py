import socket
import threading

def setup_server():
    HOST = 'localhost'
    PORT = 65432
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen()
    return server, HOST, PORT

def handle_client(conn, addr, client_id, start_event, print_lock, client_data, data_lock):
    with conn:
        with print_lock:
            print(f"[CONNECTED] {addr} as Client {client_id}")

        # Wait until start_event is set (i.e. 2 clients are connected)
        start_event.wait()

        # After start_event is released, flush the initial data
        with data_lock:
            if client_id in client_data:
                client_data[client_id] = []

        while True:
            try:
                data = conn.recv(1024).decode().strip()
                if not data:
                    with print_lock:
                        print(f"[DISCONNECTED] Client {client_id} at {addr} (Graceful)")
                    break

                with data_lock:
                    if client_id not in client_data:
                        client_data[client_id] = []
                    client_data[client_id].append(data)

                with print_lock:
                    print(f"[Client {client_id} | {addr}] Received: {data}")

            except (ConnectionResetError, ConnectionAbortedError):
                with print_lock:
                    print(f"[DISCONNECTED] Client {client_id} at {addr} (Error)")
                break

def start_server():
    server, HOST, PORT = setup_server()

    print_lock = threading.Lock()
    client_lock = threading.Lock()
    data_lock = threading.Lock()
    client_data = {}
    client_threads = []
    client_count = 0
    required_clients = 2
    start_event = threading.Event()  # Used to release clients simultaneously

    print(f"[STARTING] Server is listening on {HOST}:{PORT}")
    server.settimeout(0.1)

    try:
        while True:
            try:
                conn, addr = server.accept()

                with client_lock:
                    client_count += 1
                    client_id = client_count

                thread = threading.Thread(
                    target=handle_client,
                    args=(conn, addr, client_id, start_event, print_lock, client_data, data_lock),
                    daemon=True
                )
                thread.start()
                client_threads.append(thread)

                if client_count == required_clients:
                    print(f"[READY] {required_clients} clients connected. Starting message processing...")
                    start_event.set()  # Signal all clients to begin receiving messages

            except socket.timeout:
                continue

    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Server stopped by user.")
        server.close()

        # Print all client data received
        print("\n[CLIENT DATA DUMP]")
        with data_lock:
            for cid, messages in client_data.items():
                print(f"Client {cid}: {messages}")

if __name__ == "__main__":
    start_server()

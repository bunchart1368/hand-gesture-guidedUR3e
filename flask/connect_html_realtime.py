from flask import Flask, render_template, jsonify
import socket
import threading

app = Flask(__name__)
latest_command = None

def receive_data():
    global latest_command
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('localhost', 12345))
    
    while True:
        data = client_socket.recv(1024).decode()
        print(f"Latest Command Updated: {latest_command}")

        if not data:
            break
        latest_command = data  # Store latest command
        print(f"Received Command: {data}")
    
    client_socket.close()

@app.route('/')
def index():
    return render_template('index.html', latest_command=latest_command)

@app.route('/get_command', methods=['GET'])
def get_command():
    return jsonify({'latest_command': latest_command})

if __name__ == '__main__':
    threading.Thread(target=receive_data, daemon=True).start()
    app.run(debug=True, port=5000)

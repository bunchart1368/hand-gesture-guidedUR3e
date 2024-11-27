import socket
import re

def server_connection():
    global client_socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('localhost', 12345))
    print("Connected to the server.")
    
def get_from_server():
    data = client_socket.recv(1024).decode()
    print("Received from server: ", data)
    return data

def extract_last_tuple(s):
    # Find all tuples in the string, including those with floating-point numbers
    tuples = re.findall(r'\(-?\d+\.?\d*,-?\d+\.?\d*\)', s)
    # Return the last tuple if any are found
    if tuples:
        last_tuple_str = tuples[-1]
        # Remove parentheses and split by ','
        last_tuple = last_tuple_str.strip('()').split(',')
        # Convert to floats and return as a tuple
        return tuple(map(float, last_tuple))
    return None

def compute_target_pose(prev_pose, position, command, scale_factor=0.000009):
    """
    Computes the target pose for the robot based on the previous pose and the current position input.

    Args:
        prev_pose (list): Previous position of the robot [x, y].
        position (float): Position received from the server to adjust movement.
        scale_factor (float): Scaling factor for converting input to real-world measurements.

    Returns:
        list: Target position [x, y].
    """
    target_pose = prev_pose[:]
    # Convert server input to meters and add to previous pose
    position = int(position)
    if command == 1:
        position = -25 + (25 - (-25)) * (position - 10) / (60 - 10)
        print('Enter first command')
        print('Amplitude: ', position)
    elif command == 2:
        position = -30 + (30 - (-30)) * (position - 10) / (50 - 5)
        print('Enter second command')
        print('Amplitude: ', position)
    elif command == 3:
        position = -30 + (30 - (-30)) * (position - 10) / (60 - 25)
        print('Enter third command')
        print('Amplitude: ', position)
    else:
        position = -50 + (50 - (-50)) * (position - 10) / (60 - 10)
    target_pose[0] += int(position) * scale_factor  # Modify x based on input
    return target_pose

if __name__ == '__main__':
    server_connection()
    prev_pose = [0, 0]  # Initialize previous pose
    try:
        while True:
            data = get_from_server()
            command, position = extract_last_tuple(data)
            print(f"Command: {command}, Position: {position}")
            if isinstance(position, (int, float)):  # Ensure valid numeric input
                target_pose = compute_target_pose(prev_pose, position, command)
                print(f"Target Pose: {target_pose}")
                prev_pose = target_pose  # Update previous pose
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        client_socket.close()
        print("Client connection closed.")
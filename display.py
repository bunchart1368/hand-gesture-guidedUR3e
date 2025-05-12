import time
import json

def get_robot_data():
    try:
        with open("./final-project-1/robot_state.json", "r") as f:
            data = json.load(f)
            return {
                "command": data.get("command", 0),
                "speed": data.get("speed", 0.0),
                "force": tuple(data.get("force", (0, 0, 0))),
                "torque": tuple(data.get("torque", (0, 0, 0))),
                "position": tuple(data.get("position", (0, 0, 0))),
                "foot_pedal": data.get("foot_pedal", False),
                "depth_estimation": data.get("depth_estimation", False)
            }
    except:
        return {
            "command": 0,
            "speed": 0.0,
            "force": (0, 0, 0),
            "torque": (0, 0, 0),
            "position": (0, 0, 0),
            "foot_pedal": False,
            "depth_estimation": False
        }


while True:
    command, speed, force, torque, position, foot_pedal, depth_estimation = get_robot_data().values()
    print(f"Position: {position}")
    print(f"Force: {force}")
    print(f"Torque: {torque}")
    print(f"Command: {command}")
    print(f"Speed: {speed}")
    print(f"Foot Pedal: {foot_pedal}")
    print(f"Depth Estimation: {depth_estimation}")

    # Add a delay or break condition to avoid infinite loop in a real scenario
    time.sleep(0.1)  # Sleep for 1 second before the next update

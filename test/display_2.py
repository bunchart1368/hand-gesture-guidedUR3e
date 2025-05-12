import sys
import cv2
import numpy as np
import json
import time
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# ----- Thread: Camera -----
class CameraThread(QThread):
    change_pixmap = pyqtSignal(QImage)

    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame = cv2.flip(frame, 0)

            height, width, _ = frame.shape
            mid_x, mid_y = width // 2, height // 2

            cv2.circle(frame, (mid_x, mid_y), 5, (0, 255, 0), -1)
            cv2.circle(frame, (mid_x, mid_y), 100, (255, 0, 0), 2)
            cv2.line(frame, (0, mid_y), (width, mid_y), (0, 255, 255), 1)
            cv2.line(frame, (mid_x, 0), (mid_x, height), (0, 255, 255), 1)

            text = f"Resolution: {width}x{height}"
            center_coords = f"Center: ({mid_x}, {mid_y})"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, center_coords, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qt_img = QImage(rgb_image.data, width, height, 3 * width, QImage.Format_RGB888)
            self.change_pixmap.emit(qt_img)

        cap.release()

# ----- Thread: Robot Data -----
class RobotDataThread(QThread):
    update_data = pyqtSignal(dict)

    def run(self):
        while True:
            try:
                with open("./final-project-1/robot_state.json", "r") as f:
                    data = json.load(f)
                    robot_data = {
                        "command": data.get("command", 0),
                        "speed": data.get("speed", 0.0),
                        "force": tuple(data.get("force", (0, 0, 0))),
                        "torque": tuple(data.get("torque", (0, 0, 0))),
                        "position": tuple(data.get("position", (0, 0, 0))),
                        "foot_pedal": data.get("foot_pedal", False),
                        "depth_estimation": data.get("depth_estimation", False)
                    }
            except:
                robot_data = {
                    "command": 0,
                    "speed": 0.0,
                    "force": (0, 0, 0),
                    "torque": (0, 0, 0),
                    "position": (0, 0, 0),
                    "foot_pedal": False,
                    "depth_estimation": False
                }

            self.update_data.emit(robot_data)
            time.sleep(0.1)

# ----- Main Window -----
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Robot & Camera Dashboard")

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(640, 480)

        # Robot Data Labels
        self.command_label = QLabel("Command: 0")
        self.speed_label = QLabel("Speed: 0.0")
        self.position_label = QLabel("Position: (0, 0, 0)")
        self.force_label = QLabel("Force: (0, 0, 0)")
        self.torque_label = QLabel("Torque: (0, 0, 0)")
        self.foot_pedal_label = QLabel("Foot Pedal: False")
        self.depth_estimation_label = QLabel("Depth Estimation: False")

        self.data_labels = [
            self.command_label,
            self.speed_label,
            self.position_label,
            self.force_label,
            self.torque_label,
            self.foot_pedal_label,
            self.depth_estimation_label
        ]

        for label in self.data_labels:
            label.setStyleSheet("font-size: 16px;")

        # Layout for data
        data_layout = QVBoxLayout()
        for label in self.data_labels:
            data_layout.addWidget(label)

        # Main layout
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.image_label)
        main_layout.addLayout(data_layout)

        self.setLayout(main_layout)

        # Start camera thread
        self.cam_thread = CameraThread()
        self.cam_thread.change_pixmap.connect(self.update_image)
        self.cam_thread.start()

        # Start robot data thread
        self.data_thread = RobotDataThread()
        self.data_thread.update_data.connect(self.update_robot_data)
        self.data_thread.start()

    def update_image(self, qt_img):
        self.image_label.setPixmap(QPixmap.fromImage(qt_img))

    def update_robot_data(self, data):
        self.command_label.setText(f"Command: {data['command']}")
        self.speed_label.setText(f"Speed: {data['speed']:.2f}")
        self.position_label.setText(f"Position: {data['position']}")
        self.force_label.setText(f"Force: {data['force']}")
        self.torque_label.setText(f"Torque: {data['torque']}")
        self.foot_pedal_label.setText(f"Foot Pedal: {data['foot_pedal']}")
        self.depth_estimation_label.setText(f"Depth Estimation: {data['depth_estimation']}")

# ----- App Runner -----
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

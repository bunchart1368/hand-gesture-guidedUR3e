# Hand-Gesture-Guided Robotic Laparoscope for MIS

> **Senior Project — Robotics & AI Engineering, Chulalongkorn University, Thailand**

A collaboration between engineering and medicine to develop a **robotic system guided by hand gestures** for **laparoscope manipulation** in **Minimally Invasive Surgery (MIS)**. The system lets surgeons adjust the camera’s tilt and orientation using intuitive, sterile, pinky-based gestures while a UR robotic arm executes smooth, constrained motions around the trocar entry point.

⚡ **Key Highlights:**

* **Pinky-only gestures**: 4 intuitive camera commands (Tilt Up/Down/Left/Right)
* **UR3e robotic arm** with dynamic TCP to ensure safe entry-point pivoting
* **Emergency stop**: YOLOv11 organ detection + Endo-Depth model for depth estimation

---

## 📑 Table of Contents

* [🚀 Project Overview](#-project-overview)
* [✋ Hand-Gesture Recognition](#-hand-gesture-recognition)
* [🛑 Emergency Stop System](#-emergency-stop-system)
* [🤖 Hardware & Robot Control](#-hardware--robot-control)
* [▶️ Usage](#️-usage)
* [📂 Repository Structure](#-repository-structure)
* [🙏 Acknowledgments](#-acknowledgments)

---

## 🚀 Project Overview

The goal is to **reduce assistant workload**, **improve camera stability**, and **streamline surgeon workflow** in MIS by enabling **contactless camera control**.

**System components:**

* MediaPipe hand tracking
* Machine learning classifiers (DT, KNN, LR, NN)
* UR3e robot control via RTDE
* Dynamic TCP calculation
* YOLOv11 detection + Endo-Depth depth estimation

---

## ✋ Hand-Gesture Recognition

**Gesture Set (Version 3):**

| Right Hand | Left Hand  | Command    |
| ---------: | :--------- | :--------- |
|      Pinky | Pinky      | Tilt Up    |
|      Pinky | No command | Tilt Right |
| No command | Pinky      | Tilt Left  |
| No command | No command | Tilt Down  |

**Data Collection:**

* **Real-time capture** with MediaPipe landmarks
* **Video frame extraction** (\~885 frames/participant)
* Two batches: Model selection (1080 samples) & production dataset (5982 samples)

**Models:** Decision Tree, KNN, Logistic Regression, Neural Network

* All >90% accuracy
* Exported as `.pkl` (sklearn pipelines) & `.tflite` (NN)

---

## 🛑 Emergency Stop System

### YOLOv11 (Custom-Trained)

* Precision: 0.9986
* Recall: 1.0
* mAP\@50: 0.995

### Depth Estimation

* Compared **MiDaS vs Endo-Depth**
* **Endo-Depth** chosen (lower MAE, stable under surgical lighting)
* Organ-specific regression fits (Logarithmic for liver, Quadratic for abdominal wall)

**Final Pipeline:** YOLOv11 detects organs → Endo-Depth estimates depth → regression calibration → threshold check → **E-stop trigger**.

---

## 🤖 Hardware & Robot Control

**Supported Commands:** Up, Down, Left, Right (tool-frame rotations, pivot-safe)

**Dynamic TCP Setup:**

1. Place laparoscope tip on surface (initial TCP)
2. Free-drive for view optimization
3. Press pedal → calculate offset `d` → update TCP = 421 mm − d

**Safety:**

* Pedal/Alt key → motion only when pressed
* Boundary box defined in `robot_variables.yml`

---

## ▶️ Usage

1. **Connect UR** and set initial TCP at the laparoscope tip.
2. Run the main application:

   ```bash
   python main.py
   ```
3. Use free-drive mode to optimize the viewing angle.
4. Hold the foot pedal (or Alt key) and perform pinky/no-command gestures to tilt the camera.
5. Emergency stop triggers automatically if proximity thresholds are exceeded.

---

## 📂 Repository Structure

```
URBasic/                      # UR utilities / drivers
flask/                        # Flask app for testing/demo
hand-gesture-recognition/     # ML models & pipelines
ur_log/                       # Runtime logs
main.py                       # Entry point script
gripper.py                    # Gripper control
kk_keypoint.csv               # Sample gesture dataset
robot_variables.yml           # Config (IP, TCP, boundaries)
```

---

## 🙏 Acknowledgments

* **Faculty of Medicine, Chulalongkorn University** — special thanks to **Dr. Sopark Manasnayakorn** & **Dr. Voranaddha Vacharathit**
* MediaPipe, UR RTDE, YOLOv11, Endo-Depth
* NI Vision Builder AI for calibration

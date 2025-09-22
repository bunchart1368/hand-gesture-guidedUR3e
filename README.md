# Hand‑Gesture‑Guided Robotic Laparoscope for MIS

> Senior project — Robotics & AI Engineering, Chulalongkorn University, Thailand

A collaboration between engineering and medicine to develop a **robotic system guided by hand gestures** for **laparoscope manipulation** in **Minimally Invasive Surgery (MIS)**. The system lets surgeons adjust the camera’s tilt and orientation using intuitive, sterile, pinky‑based gestures while a UR robotic arm executes smooth, constrained motions around the trocar entry point. Safety features include a pedal‑to‑enable mechanism, workspace boundaries, dynamic TCP, and an emergency‑stop pipeline powered by organ detection and depth estimation.

---

## Table of Contents

* [Project Overview](#project-overview)
* [Key Features](#key-features)
* [System Architecture](#system-architecture)
* [Getting Started](#getting-started)
* [Repository Layout](#repository-layout)
* [Usage](#usage)
* [Surgeon Consultation](#surgeon-consultation)

  * [Initial Consultation](#initial-consultation)
  * [Re‑Consultation with Prototype](#re-consultation-with-prototype)
  * [Design Implications](#design-implications)
* [Hand‑Gesture Recognition](#hand-gesture-recognition)

  * [Gesture Set (Version 3)](#gesture-set-version-3)
  * [Data Collection](#data-collection)
  * [Batches & Splits](#batches--splits)
  * [Models & Pipelines](#models--pipelines)
  * [Performance Summary](#performance-summary)
* [Emergency Stop via Detection + Depth](#emergency-stop-via-detection--depth)

  * [Custom YOLOv11](#custom-yolov11)
  * [Depth Evaluation & Model Choice](#depth-evaluation--model-choice)
  * [Organ‑Specific Calibration](#organ-specific-calibration)
  * [Final Pipeline](#final-pipeline)
* [Hardware & Robot Control](#hardware--robot-control)

  * [Robot Movements](#robot-movements)
  * [Dynamic TCP Setup](#dynamic-tcp-setup)
  * [Safety Mechanisms](#safety-mechanisms)
  * [Force/Torque‑Based TCP Evaluation](#forcetorque-based-tcp-evaluation)
  * [Laparoscope Mount](#laparoscope-mount)
* [Citations & Acknowledgments](#citations--acknowledgments)
* [License](#license)

---

## Project Overview

The goal is to **reduce assistant workload**, **improve camera stability**, and **streamline surgeon workflow** in MIS by enabling **contactless camera control**. The project integrates:

* **MediaPipe‑based** hand landmark tracking
* **Machine‑learning classifiers** for pinky/no‑command gestures
* **UR3e robotic arm** control via RTDE (Ethernet)
* **Dynamic Tool Center Point (TCP)** to preserve entry‑point pivoting
* **Foot‑pedal deadman switch** (press‑to‑move)
* **Emergency stop** using **YOLOv11** organ detection + **Endo‑Depth** depth estimation

> The system is designed to be **simple, intuitive, and compatible with surgical workflow**, with controls limited to clinically relevant camera motions.

## Key Features

* **Pinky‑only, dual‑hand gestures** → 4 intuitive camera commands (Tilt Up/Down/Left/Right)
* **Real‑time UR control** with tool‑frame rotations (preserve trocar as pivot)
* **Dynamic TCP** computed at setup for collision‑safe tilting
* **Safety layer**: pedal‑to‑enable, workspace boundary box, E‑stop on proximity
* **Model suite**: DT, KNN, LR, and a lightweight NN; pipelines saved as **.pkl**/**TFLite**

## System Architecture

```
Hands → MediaPipe landmarks → Gesture classifier → Command mapper
      → UR3e (RTDE) motion (tool‑frame X/Y tilt) with dynamic TCP + boundaries
      → (Parallel) YOLOv11 detection + Endo‑Depth → Proximity check → E‑stop
```

## Getting Started

### Prerequisites

* Python 3.9+
* UR3e (or compatible) with RTDE enabled (URCaps/Polyscope 5.x+)
* GPU recommended for training/inference (YOLO/Depth)
* Recommended Python packages: `mediapipe`, `opencv-python`, `numpy`, `scikit-learn`, `tensorflow`/`tflite-runtime`, `ultralytics` (or equivalent YOLOv11 lib), `pyyaml`, `ur-rtde`

### Installation

```bash
# clone
git clone https://github.com/<your-org-or-user>/<your-repo>.git
cd <your-repo>

# (optional) create env
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# install deps
pip install -r requirements.txt
```

> Add or edit `requirements.txt` based on your local modules.

### Configuration

* **Robot IP & boundaries** in `robot_variables.yml`
* **Model paths** (gesture .pkl / NN .tflite, YOLO weights, Endo‑Depth) in your config section
* **Foot pedal / key binding** (default: `Alt` as enable)

## Repository Layout

> (Snapshot based on current repo; adjust as needed.)

```
URBasic/                      # UR utilities / drivers
flask/                        # (Optional) web/visualization service
hand-gesture-recognition/     # ML pipelines, training scripts, saved models
ur_log/                       # Runtime logs
figures/                      # Images for documentation/README
.gitignore
gripper.py
kk_keypoint.csv               # Example gesture dataset
main.py                       # Entry point / orchestrator
robot_variables.yml           # Robot IP, boundaries, TCP params, etc.
```

## Usage

1. **Connect UR** and set initial TCP at the **laparoscope tip on surface**.
2. **Run app** to start tracking + robot control:

   ```bash
   python main.py
   ```
3. **Free‑drive** to optimize view → **press foot pedal** to enable motion + start gestures.
4. Perform **pinky/no‑command** gestures (both hands) to tilt camera as needed.
5. E‑stop triggers automatically if proximity threshold is exceeded.

---

## Surgeon Consultation

The project began with interviews and surveys with surgeons from the **Department of Surgery, Faculty of Medicine, Chulalongkorn University**, and observation of a real MIS case at **Chulalongkorn Hospital** (see `figures/chula-medicine.jpg`).

### Initial Consultation

* Maintaining **stable laparoscope control** is a key pain point.
* Desire for an **intuitive**, low‑strain method that fits surgical flow.

### Re‑Consultation with Prototype

Meetings with **Dr. Sopark Manasnayakorn** and **Dr. Voranaddha Vacharathit** provided usability feedback on the prototype (see `figures/chula-tools.jpg`).

### Design Implications

* **Ease & Intuition:** Controls must be learnable in minutes, low cognitive load.
* **Pinky‑only Input:** Pinky remains free while holding tools → ideal for gestures.
* **Scoped Motions:** Restrict to **X/Y translation (as tool‑frame tilts)** and **Z‑axis rotation** when needed to avoid workflow disruption.

---

## Hand‑Gesture Recognition

### Gesture Set (Version 3)

Two states per hand: **`pinky`** and **`no command`** → combined into 4 commands:

| Right Hand | Left Hand  | Command    |
| ---------: | :--------- | :--------- |
|      Pinky | Pinky      | Tilt up    |
|      Pinky | No command | Tilt right |
| No command | Pinky      | Tilt left  |
| No command | No command | Tilt down  |

See `figures/handcommand.png` for visual examples.

### Data Collection

* **Format:** 21 (x,y) landmarks from MediaPipe, normalized to wrist distance; labels by human annotators.
* **Methods:** (1) **Real‑time capture** during tasks (see `figures/data collect real-time capture.png`), (2) **Video frame extraction** from recorded sessions (see `figures/datacollect_record.png`).
* **Videos:** 3 participants (\~885 frames each), sampling every 15 frames; labels: `1`=pinky, `0`=other, `None`=not visible; saved to CSV.

### Batches & Splits

* **Batch 1 — Model Selection** (real‑time): 3 participants; **1080** samples (565 pinky / 515 no‑command).
* **Batch 2 — Production Training**: diverse lighting/orientation; mixed capture modes across 4+ participants; **5982** samples (per‑hand breakdown in tables/images).
* **Train/Test:** typically **70/30** split; per‑hand evaluation when applicable.

### Models & Pipelines

* **DT** (criterion=gini, max\_depth=7, min\_samples\_split=6; grid‑searched)
* **KNN** (MinMax, k=9, euclidean; grid‑searched over k and distance)
* **LR** (20 selected landmarks, MinMax, `saga`, OvR; features chosen by correlation to minimize multicollinearity)
* **NN** (42‑dim input; Dropout(0.2)→Dense(20,ReLU)→Dropout(0.4)→Dense(10,ReLU)→Dense(2,Softmax); early stopping & checkpoint; exported to **TFLite**)

> DT visualization highlights importance of **LM17‑y (pinky base)**, **LM4‑x (thumb tip)**, **LM19‑y (pinky tip)** (see `DT top4.png`).

### Performance Summary

All models exceeded **90%** accuracy on held‑out test splits (Batch 1). Example metrics:

| Model               | Acc  | Prec | Rec  | F1   |
| ------------------- | ---- | ---- | ---- | ---- |
| Neural Network      | 0.94 | 0.90 | 1.00 | 0.95 |
| Decision Tree       | 0.98 | 0.97 | 0.97 | 0.97 |
| KNN                 | 0.99 | 0.99 | 0.98 | 0.99 |
| Logistic Regression | 0.94 | 0.97 | 0.91 | 0.94 |

External video testing (Batch 2) confirmed robustness across participants and capture modes; see confusion matrices/plots in the report figures.

---

## Emergency Stop via Detection + Depth

### Custom YOLOv11

Trained on images from our laparoscopic setup (abdominal wall, gallbladder, liver). Achieved near‑perfect detection:

* **Precision (bbox):** 0.9986
* **Recall (bbox):** 1.0000
* **mAP\@50:** 0.995
* **mAP\@50‑95:** 0.9428
* **Fitness:** 0.9480

See `figures/yolo_gal.jpg`.

### Depth Evaluation & Model Choice

We compared **MiDaS (Small/Hybrid/Large)** vs **Endo‑Depth** using a controlled height‑varying rig with ground truth from NI Vision Builder AI (see `figures/depth_eval.png`). Results showed **systematic bias** and larger MAE for MiDaS at close range; **Endo‑Depth** produced **lower MAE and bias** across ranges and organs.

### Organ‑Specific Calibration

To improve metric depth→distance, we fit per‑organ regressions:

* **Liver:** logarithmic fit; MAE ≈ **0.90 cm**, R² ≈ **0.923**
* **Abdominal wall:** quadratic fit; MAE improvement up to **28.6%** over raw Endo‑Depth in 10–35 cm range

See `figures/output.png` and summary table in the report.

### Final Pipeline

Per frame: **YOLOv11** detects targets → **Endo‑Depth** yields pixel‑wise depths → organ‑specific mapping → **threshold check** → **E‑stop** if proximity is too close. See sample output `figures/predicted_depth.png`.

---

## Hardware & Robot Control

### Robot Movements

Four surgeon‑facing commands mapped to **tool‑frame rotations** that preserve the trocar pivot:

* **Down:** −Rx (X‑axis rotation)
* **Up:** +Rx
* **Left:** −Ry (Y‑axis rotation)
* **Right:** +Ry

See `figures/Laparoscope_coordination.png`.

### Dynamic TCP Setup

1. Place laparoscope **tip on the surface** (initial TCP).
2. **Free‑drive** to optimize view; press **foot pedal** to arm hand‑tracking & depth.
3. Compute offset **d** between surface TCP and current tip position; update working TCP to **(421 mm − d)**.

This keeps pitch/yaw motions **collision‑safe** around the entry point. See `figures/TCP1.png`, `figures/TCP2.png`, `figures/TCP3.png`.

### Safety Mechanisms

* **Foot pedal / `Alt` key** (deadman): motion only while pressed (hardware pedal or keyboard)
* **Boundary box**: workspace limits set in `robot_variables.yml`; defined relative to the initial TCP/surface

### Force/Torque‑Based TCP Evaluation

Used UR3e’s built‑in **force/torque sensor** (`get_tcp_force`) to validate TCP accuracy under motions (up/down/left/right) across offsets (e.g., 5 cm, 7.5 cm, 10 cm) with/without an acrylic guide fixture. Forces typically **5–10 N**; torque trends reveal sensitivity to orientation and fixture thickness. Figures: `FT_5_2.png`, `FT_75_2.png`, `FT_10_2.png`, `FT_Without box_2.png`.

**Takeaway:** Current TCP is close to true value but can be further refined; fixture thickness and small clearances influence torque fluctuations. Adjust offsets/fixture to improve fidelity.

### Laparoscope Mount

Custom **3D‑printed** end‑effector mount secured by four bolts + clamp screw; designed to withstand gripper forces and ensure camera stability (see `figures/mounted_section2.png`).

---

## Citations & Acknowledgments

* **MediaPipe** for hand landmarks
* **UR RTDE** for real‑time control
* **YOLOv11** for detection; **Endo‑Depth** (Recasens *et al.*) for endoscopic depth
* **NI Vision Builder AI** for calibration ground truth
* Department of Surgery, Faculty of Medicine, **Chulalongkorn University** — special thanks to **Dr. Sopark Manasnayakorn** and **Dr. Voranaddha Vacharathit** for domain guidance and feedback

> Include full BibTeX/links in a `references.bib` or `CITATIONS.md` as needed.

## License

Choose a license (e.g., MIT/Apache‑2.0). Add `LICENSE` file and update this section.

---

### Maintainers

* Team **bunchart1368** — Robotics & AI Engineering, Chulalongkorn University

> For questions or demos, open an issue or contact the maintainers.

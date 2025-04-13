
# 📦 Depth Safety System 

This system uses real-time **depth estimation** and **object detection** to ensure safety during robotic-assisted procedures. It monitors the distance between surgical tools (like a laparoscope) and organs, and stops the robot if it gets too close to a critical structure.

---

## 🧠 How it works

- Uses a pretrained **YOLOv8 model** to detect organs
- Uses a **stereo depth model (Endo-Depth)** to estimate how far the organs are from the camera
- If an object is detected **too close**, the system:
  - Shows a visual warning
  - Saves a screenshot
  - Stops the robot by returning `False` (in `depth.py`)

---

## 📁 File Overview

### `depth.py` ✅

> **Used for live robotic operation**

- Runs the real-time detection + depth pipeline
- If any organ is too close, it:
  - Saves a warning screenshot to `stop_distance/`
  - Prints a warning
  - **Returns `False` to the calling program** so the robot can stop
- Otherwise, it continues running normally and returns `True`

Use this in your main control loop like:

```python
from depth import run_depth_demo

if not run_depth_demo():
    # Stop robot or alert surgeon
    print("Too close! Robot should stop.")
```

---

### `depth_calibrate.py` ⚙️

> **Used to find the best depth threshold values before real operation**

- Helps calibrate depth thresholds for different organs
- Use this if you change the lighting, camera, or environment
- Shows you real-time distance estimates so you can decide:
  - What’s a “safe” distance?
  - When should we stop the robot?
- Once you test and find the best values, **update `ORGAN_THRESHOLDS` in `depth.py`**

---

### 🧪 How to Calibrate

1. Run:

```bash
python depth_calibrate.py
```

2. Observe the distance values shown on screen
3. Note down the closest safe value for each organ
4. Update this in `depth.py`:

```python
ORGAN_THRESHOLDS = {
    "liver": 5.4,
    "gallbladder": 5.0,
    ...
}
```

---

## 🛠️ Requirements

Make sure you've installed the following:

```bash
pip install ultralytics opencv-python torch torchvision pillow numpy
```

Also, you must have:
- `kklast.pt` (your YOLO model)
- `models/encoder.pth` and `models/depth.pth` (your Endo-Depth weights)

---

## 📸 Output

When an organ gets too close:
- A screenshot is saved to the `stop_distance/` folder
- You can check these images later to verify what triggered the stop

---

## 👩‍🔧 Tip for Integration

Use `depth.py` as a **modular safety layer** inside your UR3e or other robotic system.  
You don’t need to modify it often — just **calibrate once**, and reuse.

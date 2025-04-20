import os
import cv2
import torch
import numpy as np
from PIL import Image
from datetime import datetime
from torchvision import transforms
from ultralytics import YOLO

from apps.depth_estimate.depth_decoder import DepthDecoder
from apps.depth_estimate.resnet_encoder import ResnetEncoder


class EndoStereoDepth:
    def __init__(self, encoder_path, decoder_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.encoder = ResnetEncoder(18, False)
        self.depth_decoder = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc, scales=range(1))

        encoder_weights = torch.load(encoder_path, map_location=device)
        encoder_weights = {k: v for k, v in encoder_weights.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(encoder_weights, strict=False)

        decoder_weights = torch.load(decoder_path, map_location=device)
        self.depth_decoder.load_state_dict(decoder_weights, strict=False)

        self.encoder.to(device).eval()
        self.depth_decoder.to(device).eval()

        self.transform = transforms.Compose([
            transforms.Resize((256, 320)),
            transforms.ToTensor()
        ])

    def predict(self, image_cv2):
        input_image = Image.fromarray(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))
        input_tensor = self.transform(input_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.encoder(input_tensor)
            outputs = self.depth_decoder(features)
            depth = outputs[("disp", 0)].squeeze().cpu().numpy()

        return depth  # normalized inverse depth


def run_depth_demo():
    #custom safety thresholds per organ
    #💜💙🩵💚💛🧡❤️🩷💜💙🩵💚💛🧡❤️🩷
    #CHANGE THRESHOLDS HERE WHEN TESTING IN NEW ENVIRONMENT
    ORGAN_THRESHOLDS = {
        "gallbladder": 6.1, # 7 = 4 cm, , 6.2 = 3 cm
        "liver": 5.5, #liver 5.9 = 2 cm, 6.2 = 2.5cm, 6.3 = 3 cm
        "abdominal wall": 6.0, #6.4 = 4.5 cm,, 6.1 = 3.7 cm, 5.9 = 3 cm
        #1.5 inch 6.5 abdominal wall
        # Add more if needed
    }
    DEFAULT_THRESHOLD = 6.0  # fallback if organ not listed
    #💜💙🩵💚💛🧡❤️🩷💜💙🩵💚💛🧡❤️🩷

    encoder_path = "models/stereo/test1/epoch1/encoder.pth"
    decoder_path = "models/stereo/test1/epoch1/depth.pth"
    depth_model = EndoStereoDepth(encoder_path, decoder_path)
    yolo_model = YOLO("18aprkkyolo.pt")

    cap = cv2.VideoCapture(3)
    if not cap.isOpened():
        print("❌ Failed to open webcam.")
        return

    os.makedirs("stop_distance", exist_ok=True)
    print("[INFO] Press ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        depth_map = depth_model.predict(frame)
        depth_resized = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))

        results = yolo_model(frame)[0]

        depth_vis = cv2.normalize(depth_resized, None, 0, 255, cv2.NORM_MINMAX)
        depth_vis = depth_vis.astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)

        too_close_detected = False
        warning_info = None  # (label, timestamp)

        for box in results.boxes:
            img_h, img_w = frame.shape[:2]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1 = max(0, min(x1, img_w - 1))
            x2 = max(0, min(x2, img_w - 1))
            y1 = max(0, min(y1, img_h - 1))
            y2 = max(0, min(y2, img_h - 1))

            cls = int(box.cls[0])
            label = yolo_model.names[cls]
            patch = depth_resized[y1:y2, x1:x2]
            if patch.size == 0:
                continue

            disparity = np.median(patch)
            distance_est = 1.0 / (disparity + 1e-6)
            threshold = ORGAN_THRESHOLDS.get(label, DEFAULT_THRESHOLD)
            warning = distance_est < threshold

            color = (0, 0, 255) if warning else (0, 255, 0)
            warning_text = f"{label}: {distance_est:.1f} units"
            if warning:
                warning_text += " ⚠ TOO CLOSE"
                too_close_detected = True
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                warning_info = (label, timestamp)

            (text_w, text_h), _ = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            label_y = max(0, y1 - 10)
            text_bg_top = max(0, label_y - text_h - 10)
            text_bg_bottom = label_y
            cv2.rectangle(frame, (x1, text_bg_top), (x1 + text_w, text_bg_bottom), (0, 0, 0), -1)
            cv2.putText(frame, warning_text, (x1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        
        blended = cv2.addWeighted(frame, 0.6, depth_colored, 0.4, 0)

        cv2.imshow("Organ Detection + Depth", blended)

        if too_close_detected and warning_info:
            label, timestamp = warning_info
            filename = f"stop_distance/{label}_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"[🚨] Saved frame to {filename}")
            cap.release()
            cv2.destroyAllWindows()
            return False  # ← Return immediately if too close

        if cv2.waitKey(1) == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    return True  # No problem detected



if __name__ == "__main__":
    run_depth_demo()

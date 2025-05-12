import cv2

def main(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()

        if not ret:
            print("Error reading frame.")
            break

        display_text = f"Frame: {current_frame + 1}/{total_frames}"
        cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Video Frame Viewer', frame)
        key = cv2.waitKey(0)

        if key == ord('d'):  # next frame
            if current_frame < total_frames - 1:
                current_frame += 1
        elif key == ord('a'):  # previous frame
            if current_frame > 0:
                current_frame -= 1
        elif key == ord('c'):  # skip forward 100 frames
            current_frame = min(current_frame + 100, total_frames - 1)
        elif key == ord('z'):  # go back 100 frames
            current_frame = max(current_frame - 100, 0)
        elif key == ord('e'):  # skip forward 500 frames
            current_frame = min(current_frame + 10, total_frames - 1)
        elif key == ord('q'):  # skip forward 500 frames
            current_frame = min(current_frame - 10, total_frames - 1)
        elif key == ord('f'):  # skip forward 500 frames
            current_frame = min(current_frame + 5, total_frames - 1)
        elif key == ord('s'):  # skip forward 500 frames
            current_frame = min(current_frame - 5, total_frames - 1)

        elif key == ord('t'):  # quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_file = "evaluation/doctor_eval_video/test_kk_withfeet.MOV"  # Replace with your actual video path
    main(video_file)

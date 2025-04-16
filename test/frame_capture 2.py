import cv2

video_path = "your_video.mp4"
cap = cv2.VideoCapture(video_path)

frame_count = 0
first_action_frame = None
second_action_frame = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Video", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('1'):  # Press '1' when first action happens
        first_action_frame = frame_count
        print(f"First action detected at frame {first_action_frame}")

    if key == ord('2'):  # Press '2' when second action happens
        second_action_frame = frame_count
        print(f"Second action detected at frame {second_action_frame}")
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()

if first_action_frame is not None and second_action_frame is not None:
    frame_difference = second_action_frame - first_action_frame
    print(f"Frames between actions: {frame_difference}")

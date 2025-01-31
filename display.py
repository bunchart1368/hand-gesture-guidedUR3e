import cv2
import numpy as np

# Function to calculate the angle between two points
def calculate_angle(x, y):
    """Calculate the angle of the vector (x, y) relative to the horizontal axis."""
    angle = np.arctan2(y, x) * 180 / np.pi  # Convert radians to degrees
    return angle

# Start video capture
# cap = cv2.VideoCapture(3)


# Try macOS-specific backend
cap = cv2.VideoCapture(0 + cv2.CAP_AVFOUNDATION)

# If that doesn't work, try alternative methods
if not cap.isOpened():
    print("Failed to open with AVFoundation backend")
    cap = cv2.VideoCapture(0)  # Default method

# Verify camera connection
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print("Camera successfully connected and capturing frames")
    else:
        print("Camera connected but cannot capture frames")
else:
    print("Failed to connect to camera")

#--------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Get frame dimensions
    height, width, _ = frame.shape
    mid_x, mid_y = width // 2, height // 2

    # Draw middle point
    cv2.circle(frame, (mid_x, mid_y), 5, (0, 255, 0), -1)

    # Draw a circle around the middle point
    radius = 100
    cv2.circle(frame, (mid_x, mid_y), radius, (255, 0, 0), 2)

    # Draw horizontal and vertical lines passing through the middle point
    cv2.line(frame, (0, mid_y), (width, mid_y), (0, 255, 255), 1)
    cv2.line(frame, (mid_x, 0), (mid_x, height), (0, 255, 255), 1)

    # Calculate angles relative to horizontal and vertical axes
    angle_horizontal = calculate_angle(1, 0)  # Horizontal reference vector
    angle_vertical = calculate_angle(0, 1)    # Vertical reference vector

    # Display information
    text = f"Resolution: {width}x{height}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    center_coords = f"Center: ({mid_x}, {mid_y})"
    cv2.putText(frame, center_coords, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    horizontal_angle = f"Horizontal Angle: {angle_horizontal:.2f}°"
    cv2.putText(frame, horizontal_angle, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    vertical_angle = f"Vertical Angle: {angle_vertical:.2f}°"
    cv2.putText(frame, vertical_angle, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow("Frame", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()

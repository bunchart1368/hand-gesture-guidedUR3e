import cv2
import time
import os
from datetime import datetime

def record_video(output_dir="recordings", duration=None, camera_index=0, fps=30, resolution=(640, 480)):
    """
    Record video from an external camera
    
    Args:
        output_dir (str): Directory to save recordings
        duration (int): Recording duration in seconds (None for indefinite)
        camera_index (int): Camera device index (default 0 for first camera)
        fps (int): Frames per second
        resolution (tuple): Video resolution (width, height)
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"recording_{timestamp}.mov")
    
    # Initialize video capture
    cap = cv2.VideoCapture(camera_index)
    
    # Explicitly set color mode
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    
    # Get actual resolution (may differ from requested)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Recording at resolution: {actual_width}x{actual_height}")
    
    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # H.264 codec for .mov format
    out = cv2.VideoWriter(output_file, fourcc, fps, (actual_width, actual_height))
    
    print(f"Recording started. Press 'q' to stop.")
    start_time = time.time()
    frames_recorded = 0
    
    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to capture image.")
                break
                
            # Ensure the frame is in color (BGR format)
            if len(frame.shape) == 2 or frame.shape[2] != 3:
                print("Warning: Camera returning grayscale image. Attempting to convert to color.")
                # If grayscale, convert to color
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
            # Write the frame to output file
            out.write(frame)
            frames_recorded += 1
            
            # Display the frame
            cv2.imshow('Recording', frame)
            
            # Check for key press - wait 1ms for key and check if it's 'q'
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                print("Recording stopped by user.")
                break
                
            # Check if duration is set and has elapsed
            if duration and (time.time() - start_time) > duration:
                print(f"Recording duration of {duration} seconds reached.")
                break
                
    except KeyboardInterrupt:
        print("Recording interrupted.")
    finally:
        # Calculate actual FPS
        elapsed_time = time.time() - start_time
        actual_fps = frames_recorded / elapsed_time if elapsed_time > 0 else 0
        
        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"Recording saved to: {output_file}")
        print(f"Recorded {frames_recorded} frames in {elapsed_time:.2f} seconds (Avg FPS: {actual_fps:.2f})")
        print(f"File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    # Command line argument parsing (optional)
    import argparse
    
    parser = argparse.ArgumentParser(description="Record video from camera")
    parser.add_argument("-o", "--output", default="recordings", help="Output directory")
    parser.add_argument("-d", "--duration", type=int, help="Recording duration in seconds")
    parser.add_argument("-c", "--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("-f", "--fps", type=int, default=30, help="Frames per second (default: 30)")
    parser.add_argument("-r", "--resolution", default="1080x1920", 
                        help="Resolution in format WIDTHxHEIGHT (default: 640x480)")
    
    args = parser.parse_args()
    
    # Parse resolution
    try:
        width, height = map(int, args.resolution.split('x'))
        resolution = (width, height)
    except:
        print(f"Invalid resolution format: {args.resolution}. Using default 640x480.")
        resolution = (640, 480)
    
    # Start recording
    record_video(
        output_dir=args.output,
        duration=args.duration,
        camera_index=args.camera,
        fps=args.fps,
        resolution=resolution
    )
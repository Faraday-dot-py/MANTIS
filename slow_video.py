import cv2
import time
import os

# Create a directory to save the video
save_dir = 'captured_videos'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Get the frame width and height
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(os.path.join(save_dir, 'video.avi'), fourcc, 20.0, (frame_width, frame_height))

# Record for 10 seconds
start_time = time.time()
recording_duration = 10  # seconds

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Write the frame to the video file
        out.write(frame)
        
        # Check if the recording time is up
        elapsed_time = time.time() - start_time
        if elapsed_time > recording_duration:
            break
        
        # Optionally display the frame (for debugging)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release the webcam and video writer
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Video recording finished and saved.")

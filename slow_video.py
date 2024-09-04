import cv2
import os
import time

# Create a directory to save the images
save_dir = "captured_images"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Initialize the webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open the webcam.")
    exit()

# Set the total number of images to capture
num_images = 20
capture_interval = 0.1  # in seconds

print("Capturing images...")

for i in range(num_images):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Display the captured image
    cv2.imshow("Captured Image", frame)

    # Create the filename and save the image
    filename = os.path.join(save_dir, f"image_{i+1:02d}.jpg")
    cv2.imwrite(filename, frame)
    print(f"Saved {filename}")

    # Wait for the specified interval
    time.sleep(capture_interval)

    # Check if the user wants to quit early by pressing the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()

print("Finished capturing images.")

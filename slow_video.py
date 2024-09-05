import cv2
import os
import time

# Create a directory to save the images

save_dir = f'captured_images/l2r/capture_{len(os.listdir('captured_images/l2r'))}'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Initialize the webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open the webcam.")
    exit()

# Set the total number of images to capture
imagesPerSecond = 30
numImages = 60
stime = time.time()
captureInterval = 1/imagesPerSecond  # in seconds

print("Capturing images...")

for i in range(numImages):
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
    # time.sleep(capture_interval)

    # Check if the user wants to quit early by pressing the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# print(time.time()-stime)

# Release the webcam and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()

print("Finished capturing images.")

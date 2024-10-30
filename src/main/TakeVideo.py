import cv2
import os
import time
import keyboard

# Define the class to capture and the directories
# hello, goodbye, thank you, yes, no, i love you
name_of_class_to_capture = "hello"

# Create a directory to save the images if it doesn't exist
output_folder = f"training_data/unprocessed_training_data/{name_of_class_to_capture}"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Initialize the webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open the webcam.")
    exit()

# Set the capture parameters
fps = 15  # Frames per second to capture
num_images = 30  # Total number of frames to capture
capture_interval = 1 / fps  # Interval between frames in seconds

print("Press 'Space' to start capturing images...")

frames_left = 0
was_pressed = False
captured_frames = []  # List to store captured frames

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Display the captured image
    cv2.imshow("Captured Image", frame)

    # Check for spacebar press to start capturing
    space_pressed = keyboard.is_pressed("space")

    if space_pressed and not was_pressed:
        was_pressed = True
        frames_left = num_images
        captured_frames = []  # Reset captured frames for new sequence
        capture_sequence_folder = f'{output_folder}/capture_{len(os.listdir(output_folder))}'
        os.makedirs(capture_sequence_folder)

    elif not space_pressed:
        was_pressed = False

    # Capture images if frames are left to capture
    if frames_left > 0:
        captured_frames.append(frame)  # Store the frame in memory
        frames_left -= 1
        print(f"Captured frame {num_images - frames_left}")

        # Wait for the next frame interval
        time.sleep(capture_interval)

    # Save all frames after capturing is complete
    if frames_left == 0 and len(captured_frames) != 0:
        print("Saving sequence, please wait...")
        for i, saved_frame in enumerate(captured_frames):
            filename = f"{capture_sequence_folder}/image_{i + 1:02d}.jpg"
            cv2.imwrite(filename, saved_frame)

        captured_frames = []
        print("Sequence saved")
        print(f'You have {len(os.listdir(output_folder))} tmaps')
        print("-" * 30, end="\n\n\n")

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()

print("Finished capturing images.")

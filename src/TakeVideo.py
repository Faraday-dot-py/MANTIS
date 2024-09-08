import cv2
import os
import time
import keyboard

# Create a directory to save the images
nameOfClassToCapture = "r2l"

if not os.path.exists(f"../captured_images/{nameOfClassToCapture}"):
    os.makedirs(f"../captured_images/{nameOfClassToCapture}")

# Initialize the webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open the webcam.")
    exit()

# Set the total number of images to capture
imagesPerSecond = 30
numImages = 30
captureInterval = 1 / imagesPerSecond  # in seconds

print("Capturing images...")

framesLeft = 0
wasPressed = False
captured_frames = []  # List to store captured frames

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Display the captured image
    cv2.imshow("Captured Image", frame)

    spacePressed = keyboard.is_pressed("space")

    if spacePressed and not wasPressed:
        wasPressed = True
        framesLeft = numImages
        captured_frames = []  # Reset captured frames for new sequence
        captureSequenceFolder = f'../captured_images/{nameOfClassToCapture}/capture_{len(os.listdir(f"../captured_images/{nameOfClassToCapture}"))}'
        os.makedirs(captureSequenceFolder)

    elif not spacePressed:
        wasPressed = False

    if framesLeft > 0:
        captured_frames.append(frame)  # Store the frame in memory
        framesLeft -= 1
        print(f"Captured frame {numImages-framesLeft}")

    if framesLeft == 0 and len(captured_frames) != 0:
        print("Saving sequence, please wait")
        # Save all frames after capturing is complete
        for i, saved_frame in enumerate(captured_frames):
            filename = f"../{captureSequenceFolder}/image_{i + 1:02d}.jpg"
            cv2.imwrite(filename, saved_frame)

        captured_frames = []
        print("Sequence saved")
        print(
            f'You have {len(os.listdir(f"../captured_images/{nameOfClassToCapture}"))} tmaps'
        )
        print("-" * 30, end="\n\n\n")

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()

print("Finished capturing images.")

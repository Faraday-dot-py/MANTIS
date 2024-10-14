from lib.TemporalMap import TemporalMap

import os
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from tqdm import tqdm


# Initialize the TemporalMap object and Mediapipe Hand solutions
print("Loading tmap maker class")
tmapMaker = TemporalMap(CONTEXT_WINDOW=323)
print("Setting up tmap maker class")
tmapMaker.setup()
print("Done")

sourceDir = r"C:\Users\jacob\Documents\Github\Unnamed\training_data\unprocessed_training_data"
outputDir = r"C:\Users\jacob\Documents\Github\Unnamed\training_data\unprocessed_training_data" 
labels = os.listdir(sourceDir)  # The labels in sourceDir

# Process each label
for label in labels:
    saveDir = f"{outputDir}/{label}"
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    for video_file in os.listdir(f"{sourceDir}/{label}"):
        video_path = f"{sourceDir}/{label}/{video_file}"
        cap = cv2.VideoCapture(video_path)  # Open the video file
        handLandmarkArray = []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        with tqdm(total=total_frames, desc=f"Processing {video_file}", unit="frame") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break  # Exit the loop if there are no frames left to read

                # Convert the frame to RGB (as Mediapipe requires RGB images)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                landmarks = tmapMaker.calculateLandmarks(frame_rgb)
                timg = tmapMaker.calculateTimg(landmarks)
                tmapMaker.addTimgToBuffer(timg)

                pbar.update(1)

    # Release the video capture object
    cap.release()

    # Calculate the Tmap from the hand landmarks array
    tmap = np.array(tmapMaker.tmap)

    image_array = tmap.astype(np.uint8)  # Convert to unsigned 8-bit integer

    # Create an image object from the array
    image = Image.fromarray(image_array)

    # Display the image
    plt.imshow(image)
    plt.axis('off')
    plt.show()

    # Save the image as a png file
    image.save(f"{saveDir}/sample_{len(os.listdir(saveDir))}.png")



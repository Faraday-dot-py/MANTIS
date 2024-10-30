import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import matplotlib.animation as animation
import cv2
import os
import mediapipe as mp
import math
from concurrent.futures import ThreadPoolExecutor
import threading
import tensorflow as tf

BLANK_HAND = np.zeros((21, 3), dtype=np.uint8)
BLANK_FACE = np.zeros((468, 3), dtype=np.uint8)
BLANK_POSE = np.zeros((33, 3), dtype=np.uint8)
BLANK_IMAGE = np.concatenate((
    # BLANK_FACE,
    # BLANK_POSE, 
    BLANK_HAND, 
    BLANK_HAND
), axis=0)

mpHolistic = mp.solutions.holistic

UPPER_Z_LIMIT = 5
LOWER_Z_LIMIT = -5

class TemporalMap:
    def __init__(
        self,
        CONTEXT_WINDOW: int = 120,
        RENDER_TMAPS: bool = False,
        MODEL_PATH: str = r'C:\Users\awebb\Documents\Programming\Python\MANTIS\models\hand_landmarker.task'
    ):

        self.CONTEXT_WINDOW = CONTEXT_WINDOW
        self.RENDER_TMAPS = RENDER_TMAPS
        self.MODEL_PATH = MODEL_PATH
        self.detector = None
        self.tmap = None
        self.fig = None
        self.axs = None


    def setup(self):
        self.detector = mpHolistic.Holistic()

        self.tmap = deque([BLANK_IMAGE.copy() for _ in range(self.CONTEXT_WINDOW)], maxlen=self.CONTEXT_WINDOW)
        print(np.array(list(self.tmap)).shape)


    def convertImageToMediapipeImage(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    def calculateTimg(self, detectionResults):
        return np.concatenate((
            # detectionResults['face'],
            # detectionResults['pose'],
            detectionResults['left'],
            detectionResults['right']
        ))
    
    
    def sigmoid(self, x, k=1):
        return 1/(1 + np.exp(-x*k))
    
    def upperLowerLimit(self, x, min, max):
        return min if x < min else max if x > max else x
    
    def calculateLandmarks(self, frame, normalized=True):
        # Process the frame using the Holistic model
        results = self.detector.process(frame)

        # scalingFactor = 1 if normalized else 255
        # scalingFactor = 255

        landmarks = {
            # 'face': BLANK_FACE,
            # 'pose': BLANK_POSE,
            'left': BLANK_HAND,
            'right': BLANK_HAND
        }
        
        # if results.pose_landmarks:
        #     landmarks['pose']  = [
        #         (
        #             self.scaledSigmoid(landmark.x) * scalingFactor,
        #             self.scaledSigmoid(landmark.y) * scalingFactor, 
        #             self.sigmoid(landmark.z) * scalingFactor
        #         ) 
        #         for landmark in results.pose_landmarks.landmark]

        # if results.face_landmarks:
        #     landmarks['face']  = [
        #         (
        #             self.scaledSigmoid(landmark.x) * scalingFactor,
        #             self.scaledSigmoid(landmark.y) * scalingFactor, 
        #             self.sigmoid(landmark.z) * scalingFactor
        #         ) 
        #         for landmark in results.face_landmarks.landmark]
            
        if results.left_hand_landmarks:
            landmarks['left']  = [
                (
                    self.upperLowerLimit(landmark.x, 0, 1) * 255,
                    self.upperLowerLimit(landmark.y, 0, 1) * 255,
                    self.sigmoid(landmark.z, 1) * 255
                ) 
                for landmark in results.left_hand_landmarks.landmark]

        # if results.right_hand_landmarks:
        #     landmarks['right'] = [
        #         (
        #             self.scaledSigmoid(landmark.x),
        #             self.scaledSigmoid(landmark.y),
        #             self.scaledSigmoid(landmark.z)
        #         ) 
        #         for landmark in results.right_hand_landmarks.landmark]

        return landmarks
    

    def calculateTmapFromArray(self, detectionResultArray):
        tmap = []
        for entry in detectionResultArray:
            tmap.append(self.calculateTimg(entry))

        return tmap
    
    def addTimgToBuffer(self, timg):
        self.tmap.append(timg)



class_names = ['l2r', 'r2l']

def capture_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def process_landmarks(tmapMaker, frame):
    handLandmarks = tmapMaker.calculateLandmarks(frame)
    timg = tmapMaker.calculateTimg(handLandmarks)
    tmapMaker.addTimgToBuffer(timg)
    return tmapMaker.tmap

def makePrediction(tmapMaker):
    predictions = tmapMaker.model.predict(tmapMaker.tmap, verbose=0)
    probabilities = tf.nn.softmax(predictions[0])
    predictedClass = class_names[np.argmax(probabilities)]
    return predictedClass

if __name__ == '__main__':
    tmapMaker = TemporalMap(RENDER_TMAPS=True)
    tmapMaker.setup()

    cap = cv2.VideoCapture(0)
    plt.ion()  # Turn on interactive mode for matplotlib
    figure, ax = plt.subplots()  # Create figure and axes for plotting

    # Use a thread pool for parallel execution
    with ThreadPoolExecutor(max_workers=2) as executor:
        futureFrame = None
        futureLandmarks = None
        futurePrediciton = None

        while True:
            # Submit video capture to a thread
            futureFrame = executor.submit(capture_frame, cap)
            
            if futureLandmarks:
                # Update the plot with the processed landmarks
                tmap = futureLandmarks.result()
                ax.cla()
                ax.imshow(tmap)
                ax.set_xlabel('Hand Index')
                ax.set_ylabel('Age')
                plt.pause(0.01)

            # Wait for the captured frame
            frame = futureFrame.result()
            if frame is None:
                print("Failed to capture image, exiting.")
                break

            # Submit landmark processing to another thread
            futureLandmarks = executor.submit(process_landmarks, tmapMaker, frame)

            futurePrediction = executor.submit(makePrediction, tmapMaker)

            if futurePrediciton:
                print(futurePrediciton.result())

            cv2.imshow("Captured Image", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            # Exit the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # Release the video capture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()  # Turn off interactive mode
    plt.show()  # Ensure any remaining plots are displayed
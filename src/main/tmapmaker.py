import mediapipe as mp
import numpy as np
import cv2
from tqdm import tqdm
import copy
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
from PIL import Image

# BLANK_HAND = np.zeros((21, 3), dtype=np.uint8)
BLANK_HAND = [[0]*3]*21
mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils

class TmapMaker:
    """
    Tmap - Temporal Map
    Tmaps are a 2d representation of a pointcloud's motion through time.
    Each row of a tmap is called a timg, or Temporal Tmage.
    A timg represents a pointcloud in one point in time.
    3-dimensional cartesian coordinates are normalized and multiplied by 255 for each point, creating a row if RGB pixels.
    These rows are stacked on top of one another to form a tmap.
    Each row is 33.33 ms after the one above it.
    This is how we represent the motion of a pointcloud as a 2D image.
    """

    def __init__(self, landmarker_model_path=r'C:\Users\awebb\Documents\Programming\Python\MANTIS\src\main\hand_landmarker.task'):
        base_options = python.BaseOptions(model_asset_path=landmarker_model_path)

        options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)

        self.hand_landmarker = vision.HandLandmarker.create_from_options(options)

    """
    Takes an OpenCV image and turns it to a Mediapipe image for processing

    @param frame A cv2 image to process
    @return A mediapipe image
    """
    def convertImageToMediapipeImage(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)


    """
    Run the hand landmarker

    @param frame A mediapipe frame
    @return A mediapipe hand landmarker results object
    """
    def detectHands(self, frame):
        return self.hand_landmarker.detect(frame)
    
    def sigmoid(self, x, alpha=4, beta=-1):
        return int((1 / (1 + np.exp(-alpha*(x+beta)))) * 255)
        # return int(x if x < 1 and x > 0 else (1 if x > 1 else 0) * 255)

    """
    Generates a tmap of just the left and right hand from a Mediapipe frame

    @param frame A mediapipe frame
    @return A tmap which represents the left and right hands
    """
    def convertLandmarkerResultsToArr(self, landmarker_results, raw=False):
        left_timg = BLANK_HAND
        right_timg = BLANK_HAND

        for (hand, handedness) in zip(landmarker_results.hand_landmarks, landmarker_results.handedness):
            hand_timg = [(
                landmark.x if raw else self.sigmoid(landmark.x),
                landmark.y if raw else self.sigmoid(landmark.y),
                landmark.z if raw else self.sigmoid(landmark.z)
                
            ) for landmark in hand]

            # print(hand_timg)
            
            if handedness[0].display_name == 'Left':
                left_timg = hand_timg

            if handedness[0].display_name == 'Right':
                right_timg = hand_timg

        return np.array(left_timg + right_timg)
    
    """
    Generates a tmap based on a cv2 video

    @param videoCap a cv2 video cap object
    @param verbose whether or not to display a progress bar
    """
    def processVideo(self, videoCap, verbose=False, raw=False):
        tmap = []

        frames = int(videoCap.get(cv2.CAP_PROP_FRAME_COUNT))

        if verbose:
            with tqdm(total=frames) as pbar:
                while videoCap.isOpened():
                    ret, frame = videoCap.read()
                    if not ret:
                        break
                    
                    landmarkerResults = self.detectHands(self.convertImageToMediapipeImage(frame))

                    timg = self.convertLandmarkerResultsToArr(landmarkerResults, raw=raw)

                    tmap.append(timg)

                    pbar.update(1)

        else:
            while videoCap.isOpened():
                ret, frame = videoCap.read()

                if not ret:
                    break

                    landmarkerResults = self.detectHands(self.convertImageToMediapipeImage(frame))

                    timg = self.convertLandmarkerResultsToArr(landmarkerResults, raw=raw)

                    tmap.append(timg)

        return np.array(tmap)

    """
    Load a video from a path to a cv2 video object

    @param path The path you want to load a video from
    @return a cv2 video object
    """
    def loadVideo(self, path):
        cap = cv2.VideoCapture(path)

        return cap
    
    """
    Annotates an image with the hand landmarks
    Primarily for vision

    @param frame the frame to annotate
    @param landmarkerResults a mediapipe hand landmark results object
    @return an annotated frame with with the connections of the hand displayed
    """
    def annotateImage(self, frame, landmarkerResults):
        copiedImg = copy.copy(frame)

        if landmarkerResults.left_hand_landmarks:
            mpDraw.draw_landmarks(copiedImg, landmarkerResults.left_hand_landmarks, mpHands.HAND_CONNECTIONS)

        if landmarkerResults.right_hand_landmarks:
            mpDraw.draw_landmarks(copiedImg, landmarkerResults.right_hand_landmarks, mpHands.HAND_CONNECTIONS)

        return copiedImg


if __name__ != "__main__":
    tmapMaker = TmapMaker()
    import matplotlib.pyplot as plt
    
    # import pandas as pd
    # xs = np.linspace(-10, 10, 100)
    # ys = tmapMaker.sigmoid(xs)

    video_path = r"C:\Users\awebb\Documents\Programming\Python\MANTIS\training_data\lsa64\accept\sample_0.mp4"
    video_cap = tmapMaker.loadVideo(video_path)

    tmap = tmapMaker.processVideo(video_cap, verbose=True)

    # np.save(r"C:\Users\awebb\Documents\Programming\Python\MANTIS\training_data\lsa64_tmaps\accept\sample_0.npy", tmap)

    plt.imshow(tmap)
    plt.show()

    # plt.plot(xs, ys)
    # plt.show()

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Load the TmapMaker class
    tmapMaker = TmapMaker()


    # Display the tmap
    # import matplotlib.pyplot as plt
    # plt.imshow(tmap)
    # plt.show()

    tld_input = r'C:\Users\awebb\Documents\Programming\Python\MANTIS\training_data\lsa64'
    tld_output = r'C:\Users\awebb\Documents\Programming\Python\MANTIS\training_data\lsa64_tmaps'

    for video_class in os.listdir(tld_input):
        video_class_path = os.path.join(tld_input, video_class)
        video_class_output_path = os.path.join(tld_output, video_class)

        if not os.path.exists(video_class_output_path):
            os.makedirs(video_class_output_path)

        for video in os.listdir(video_class_path):
            video_path = os.path.join(video_class_path, video)
            video_output_path = os.path.join(video_class_output_path, video.replace('.mp4', '.jpg'))

            video_cap = tmapMaker.loadVideo(video_path)

            tmap = tmapMaker.processVideo(video_cap, verbose=True, raw=True)

            tmap_img = Image.fromarray((tmap * 255).astype(np.uint8))

            tmap_img.save(video_output_path)
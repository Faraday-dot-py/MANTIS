import mediapipe as mp
import numpy as np
import cv2
from tqdm import tqdm

# BLANK_HAND = np.zeros((21, 3), dtype=np.uint8)
BLANK_HAND = [[0]*3]*21

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

    def __init__(self):
        self.handLandmarker = mp.solutions.holistic.Holistic()

    """
    Takes an OpenCV image and turns it to a Mediapipe image for processing

    @param frame A cv2 image to process
    @return A mediapipe image
    """
    def convertImageToMediapipeImage(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    """
    Generates a tmap of just the left and right hand from a Mediapipe frame

    @param frame A mediapipe frame
    @return A tmap which represents the left and right hands
    """
    def processFrame(self, frame):
        mpImg = self.convertImageToMediapipeImage(frame)

        landmarkerResults = self.handLandmarker.process(frame)

        leftHand = BLANK_HAND
        rightHand = BLANK_HAND

        if landmarkerResults.left_hand_landmarks:
            leftHand = [[landmark.x, landmark.y, landmark.z] for landmark in landmarkerResults.left_hand_landmarks.landmark]

        if landmarkerResults.right_hand_landmarks:
            rightHand = [[landmark.x, landmark.y, landmark.z] for landmark in landmarkerResults.right_hand_landmarks.landmark]

        
        return leftHand + rightHand
    
    """
    Generates a tmap based on a cv2 video

    @param videoCap a cv2 video cap object
    @param verbose whether or not to display a progress bar
    """
    def processVideo(self, videoCap, verbose=False):
        tmap = []

        frames = int(videoCap.get(cv2.CAP_PROP_FRAME_COUNT))

        if verbose:
            with tqdm(total=frames) as pbar:
                while videoCap.isOpened():
                    ret, frame = videoCap.read()
                    if not ret:
                        break

                    timg = self.processFrame(frame)

                    tmap.append(timg)

                    pbar.update(1)

        else:
            while videoCap.isOpened():
                ret, frame = videoCap.read()

                if not ret:
                    break

                timg = self.processFrame(frame)

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
    

if __name__ == "__main__":
    # Load the TmapMaker class
    tmapMaker = TmapMaker()

    # Load the video you want to process
    video = tmapMaker.loadVideo(r'C:\Users\awebb\Documents\Programming\Python\MANTIS\testing\ASL_Videos\Blabbermouth.mp4')

    # Process the video into a tmap
    tmap = tmapMaker.processVideo(video, verbose=True)

    # Display the tmap
    import matplotlib.pyplot as plt

    plt.imshow(tmap)
    plt.show()

    
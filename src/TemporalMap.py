import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import matplotlib.animation as animation
import cv2
import os

BLANK_IMAGE = np.zeros((60, 42, 3), dtype=np.uint8)
BLANK_LINE = np.zeros((21, 3), dtype=np.uint8)


class TemporalMap:
    def __init__(
        self,
        CONTEXT_WINDOW: int = 60,
        RENDER_TMAPS: bool = False,
        DISPLAY_CAMERA_VIEW: bool = False,
        VIDEO_CAP_INDEX: int = 0,
        MODEL_PATH: str = r'C:\Users\awebb\Documents\Programming\Python\Unnamed\models\hand_landmarker.task'
    ):

        self.CONTEXT_WINDOW = CONTEXT_WINDOW
        self.RENDER_TMAPS = RENDER_TMAPS
        self.DISPLAY_CAMERA_VIEW = DISPLAY_CAMERA_VIEW
        self.MODEL_PATH = MODEL_PATH
        self.detector = None
        self.tmap = None
        self.cap = cv2.VideoCapture(VIDEO_CAP_INDEX)
        self.fig = None
        self.axs = None

    def setup(self):
        base_options = python.BaseOptions(model_asset_path=self.MODEL_PATH)
        options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
        self.detector = vision.HandLandmarker.create_from_options(options)

        self.tmap = deque(BLANK_IMAGE, maxlen=self.CONTEXT_WINDOW)

        if self.RENDER_TMAPS:
            self.fig, self.axs = plt.subplots(1, 1, figsize=(10, 5))

    def captureImage(self):
        ret, frame = self.cap.read()

        if not ret:
            print("Error: Failed to capture image.")
            raise Exception("Camera broken :/")

        if self.DISPLAY_CAMERA_VIEW:
            cv2.imshow("Captured Image", frame)

        return frame

    def convertImageToMediapipeImage(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    def calculateHandLandmarks(self, image):
        handlms = self.detector.detect(image)
        # print(handlms)
        return handlms

    def calculateTimgPixelValues(self, detectionResult, handIndex: int):
        return [(int(landmark.x * 255), int(landmark.y * 255), int(((landmark.z + 1) / 2) * 255)) for landmark in detectionResult.hand_landmarks[handIndex]]

    def calculateTimg(self, detectionResult):
        leftHand = BLANK_LINE
        rightHand = BLANK_LINE

        if detectionResult.hand_landmarks:
            leftOrRightIsProminent = detectionResult.handedness[0][0].display_name
            
            leftHandIndex = 0 if (leftOrRightIsProminent == "Left") else 1
            rightHandIndex = int(not leftHandIndex)
            
            if len(detectionResult.handedness) == 2:
                leftHand = self.calculateTimgPixelValues(detectionResult, leftHandIndex)
                rightHand = self.calculateTimgPixelValues(detectionResult, rightHandIndex)
            
            else:
                if leftHandIndex == 0:
                    leftHand = self.calculateTimgPixelValues(detectionResult, leftHandIndex)
                else:
                    rightHand = self.calculateTimgPixelValues(detectionResult, rightHandIndex)


        timg = np.concatenate((leftHand, rightHand), axis=0)
        # if not fromArray:
        #     self.tmap.append(timg)
        
        return timg

    def calculateTmapFromArray(self, detectionResultArray):
        tmap = []
        for entry in detectionResultArray:
            tmap.append(self.calculateTimg(entry))

        return tmap

    def refreshTmap(self):
        if self.RENDER_TMAPS:
            plt.imshow(self.tmap)
            plt.draw()
            plt.pause(0.0001)
            plt.clf()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.cap.release()
            raise SystemExit()


if __name__ == '__main__':
    tmapMaker = TemporalMap(RENDER_TMAPS=True, DISPLAY_CAMERA_VIEW=True)

    tmapMaker.setup()

    while 1:
        frame = tmapMaker.captureImage()

        mpFrame = tmapMaker.convertImageToMediapipeImage(frame)

        handLandmarks = tmapMaker.calculateHandLandmarks(mpFrame)

        tmapMaker.tmap.append(tmapMaker.calculateTimg(handLandmarks))

        tmapMaker.refreshTmap()


    plt.show()

# tmapMaker = TemporalMap()

# tmapMaker.setup()

# # while 1:
# jacueblolsHand = mp.Image.create_from_file(
#     f"captured_images/r2l/capture_0/image_11.jpg"
# )

# # mpFrame = tmapMaker.convertImageToMediapipeImage(frame)

# handLandmarks = tmapMaker.calculateHandLandmarks(jacueblolsHand)

# tmapMaker.calculateTimg(handLandmarks)

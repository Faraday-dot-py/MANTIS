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

BLANK_HAND = np.zeros((21, 3), dtype=np.uint8)
BLANK_FACE = np.zeros((468, 3), dtype=np.uint8)
BLANK_POSE = np.zeros((33, 3), dtype=np.uint8)
BLANK_IMAGE = np.concatenate((BLANK_FACE, BLANK_POSE, BLANK_HAND, BLANK_HAND), axis=0)
mpHolistic = mp.solutions.holistic

class TemporalMap:
    def __init__(
        self,
        CONTEXT_WINDOW: int = 120,
        RENDER_TMAPS: bool = False,
        DISPLAY_CAMERA_VIEW: bool = False,
        VIDEO_CAP_INDEX: int = 0,
        MODEL_PATH: str = r'C:\Users\awebb\Documents\Programming\Python\MANTIS\models\hand_landmarker.task'
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
        self.detector = mpHolistic.Holistic()

        self.tmap = deque([BLANK_IMAGE.copy() for _ in range(self.CONTEXT_WINDOW)], maxlen=self.CONTEXT_WINDOW)
        print(np.array(list(self.tmap)).shape)

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

    def calculateTimgPixelValues(self, detectionResult, handIndex: int):
        return [(landmark.x, landmark.y, (landmark.z + 1)/2) for landmark in detectionResult.hand_landmarks[handIndex]]
    
    def calculateTimg(self, detectionResults):
    #     return np.concatenate((
    #         (detectionResults['face'] if detectionResults['face'] != None else BLANK_FACE),
    #         (detectionResults['pose'] if detectionResults['pose'] != None else BLANK_POSE), 
    #         (detectionResults['left'] if detectionResults['left'] != None else BLANK_HAND), 
    #         (detectionResults['right'] if detectionResults['right'] != None else BLANK_HAND)
    #         ))
        return np.concatenate((detectionResults['face'], detectionResults['pose'], detectionResults['left'], detectionResults['right']))
    
    def calculateLandmarks(self, frame):
        # Read frames from the webcam
        ret, frame = self.cap.read()
        if not ret:
            print("frames not found")
            raise SystemExit()

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame using the Holistic model
        results = self.detector.process(rgb_frame)

        landmarks = {
            'face': BLANK_FACE,
            'pose': BLANK_POSE,
            'left': BLANK_HAND,
            'right': BLANK_HAND
        }
        # print('\n-----')
        if results.pose_landmarks:
            landmarks['pose'] = [(landmark.x, landmark.y, landmark.z) for landmark in results.pose_landmarks.landmark]
            # print(np.array(landmarks['pose']).shape)
        if results.face_landmarks:
            landmarks['face'] = [(landmark.x, landmark.y, landmark.z) for landmark in results.face_landmarks.landmark]
            # print(np.array(landmarks['face']).shape)
        if results.left_hand_landmarks:
            landmarks['left'] = [(landmark.x, landmark.y, landmark.z) for landmark in results.left_hand_landmarks.landmark]
            # print(np.array(landmarks['left']).shape)
        if results.right_hand_landmarks:
            landmarks['right'] = [(landmark.x, landmark.y, landmark.z) for landmark in results.right_hand_landmarks.landmark]
            # print(np.array(landmarks['right']).shape)
        print()

        return landmarks

    def calculateHandTimg(self, detectionResult):
        leftHand = BLANK_HAND
        rightHand = BLANK_HAND

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

        handLandmarks = tmapMaker.calculateLandmarks(mpFrame)
        
        timg = tmapMaker.calculateTimg(handLandmarks)

        # print(np.array(tmapMaker.tmap).shape)
        # print(np.array(timg).shape)
        tmapMaker.tmap.append(timg)
        # # tmapMaker.tmap = timg
        # # print(np.array(timg).shape)
        # np.savetxt('out.txt', np.array(tmapMaker.tmap))

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

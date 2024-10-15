import numpy as np
import mediapipe as mp
from collections import deque

DATA_TYPE = np.uint8
BLANK_HAND = np.zeros((21, 3), dtype=DATA_TYPE)
BLANK_FACE = np.zeros((468, 3), dtype=DATA_TYPE)
BLANK_POSE = np.zeros((33, 3), dtype=DATA_TYPE)
BLANK_IMAGE = np.concatenate(
    (
        # BLANK_FACE,
        # BLANK_POSE,
        BLANK_HAND,
        BLANK_HAND,
    ),
    axis=0,
)


class TmapGenerator:
    def __init__(
        self,
        TEMPLATE_SIZE: int = 120
    ):
        self.TEMPLATE_SIZE = TEMPLATE_SIZE
        self.detector = mp.solutions.holistic.Holistic()
        self.tmap = deque(
            [BLANK_IMAGE.copy() for _ in range(self.TEMPLATE_SIZE)],
            maxlen=self.TEMPLATE_SIZE,
        )
        print("TMAP INIT: ", np.array(list(self.tmap)).shape)

    def calculateTimg(self, detectionResults):
        return np.concatenate(
            (
                detectionResults["left"],
                detectionResults["right"],
            )
        )

    def sigmoid(self, x, k=1):
        return 1 / (1 + np.exp(-x * k))

    def upperLowerLimit(self, x, min, max):
        return min if x < min else max if x > max else x

    def calculateLandmarks(self, frame):
        # Process the frame using the Holistic model
        results = self.detector.process(frame)
        landmarks = {
                    # 'face': BLANK_FACE,
                    # 'pose': BLANK_POSE,
                    "left": BLANK_HAND,
                    "right": BLANK_HAND,
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
            landmarks["left"] = [
                (
                    self.upperLowerLimit(landmark.x, 0, 1) * 255,
                    self.upperLowerLimit(landmark.y, 0, 1) * 255,
                    self.sigmoid(landmark.z, 1) * 255,
                )
                for landmark in results.left_hand_landmarks.landmark
            ]

        # if results.right_hand_landmarks:
        #     landmarks['right'] = [
        #         (
        #             self.scaledSigmoid(landmark.x),
        #             self.scaledSigmoid(landmark.y),
        #             self.scaledSigmoid(landmark.z)
        #         )
        #         for landmark in results.right_hand_landmarks.landmark]
        
        return landmarks


    def addTimgToBuffer(self, timg):
        self.tmap.append(timg)

    def process_landmarks(self, frame):
        handLandmarks = self.calculateLandmarks(frame)
        timg = self.calculateTimg(handLandmarks)
        self.addTimgToBuffer(timg)
        return self.tmap


import mediapipe as mp
import cv2
import numpy as np
from collections import deque
import os
import time


from mediapipe.tasks.python import vision
from mediapipe.tasks import python


base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)

detector = vision.HandLandmarker.create_from_options(options)


context_window = deque(maxlen=100)


camera = cv2.VideoCapture(0)


while True:
    ret, frame = camera.read()
    
    if not ret:
        print('No camera image')
        break

    # Resize the frame for better view
    # frame = cv2.resize(frame, (800, 600))

    # Convert the frame from BGR to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create a MediaPipe Image object from the numpy array
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    results = detector.detect(mp_image)

    # [(int(landmark.x*255), int(landmark.y*255),  int(((landmark.z + 1) / 2) * 255)) for landmark in detection_result.hand_landmarks[0]]

    if results.hand_landmarks:
        
        if len(results.handedness) == 2:
            # Run the code for both hands
            temporalImage = 
            pass

        else:
            lef
            
    

    time.sleep(0.5)

    # context_window.append([landmark[0].__dict__ for landmark in results.hand_landmarks])

    # print(context_window)


camera.release()







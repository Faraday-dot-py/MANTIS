
from TemporalMap import TemporalMap

import os
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import cv2


tmapMaker = TemporalMap()

tmapMaker.setup()

sourceDir = 'captured_images'
labels = ['l2r', 'r2l']

for label in labels:
    saveDir = f'output_images/{label}'
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    for sequence in os.listdir(f'{sourceDir}/{label}'):
        
        handLandmarkArray = []
        for image in os.listdir(f'{sourceDir}/{label}/{sequence}'):
            image = mp.Image.create_from_file(f'{sourceDir}/{label}/{sequence}/{image}')

            handLandmarkArray.append(tmapMaker.calculateHandLandmarks(image))

        
        tmap = np.array(tmapMaker.calculateTmapFromArray(handLandmarkArray))
        cv2.imwrite(f'{saveDir}/sample_{len(os.listdir(saveDir))}.jpg', tmap)
        

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import json
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2

IMAGE_FILES = os.listdir('./input_images')

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                    num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def dumper(obj):
    try:
        return obj.toJSON()
    except:
        return obj.__dict__

def displayResults(file_names):
    
    # Load the images
    fig, axs = plt.subplots(len(file_names), 1, figsize=(10, 5))

    for (i, file) in enumerate(file_names):

    # Display the first image on the first subplot
        axs[i].imshow(mpimg.imread(f'./output_images/{file}'))
        # axs[i].set_title(i)
        axs[i].axis('off')

    # Show the plot
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()

def load_file(file_name):
    testImage = ''
    with open(f'./output_json/{file_name}', 'r') as f:
        testImage = json.load(f)
        
    return testImage

def createTemporalImage(testImage):
    timg = HashmapObj(testImage)
    temporalImageArray = []

    # Translate the normalized values to rgb values
    for point in timg.hand_landmarks[0]:
        temporalImageArray.append((int(point['x']*255), int(point['y']*255), int(((point['z'] + 1) / 2) * 255)))

    return temporalImageArray

# Some Magic
class HashmapObj:
    def __init__(self, in_dict:dict):
        assert isinstance(in_dict, dict)
        for key, val in in_dict.items():
            if isinstance(val, (list, tuple)):
                setattr(self, key, [HashmapObj(x) if isinstance(x, dict) else x for x in val])
            else:
                setattr(self, key, HashmapObj(val) if isinstance(val, dict) else val)

def draw_landmarks_on_image(rgb_image, detection_result):
            hand_landmarks_list = detection_result.hand_landmarks
            handedness_list = detection_result.handedness
            annotated_image = np.copy(rgb_image)

            # Loop through the detected hands to visualize.
            for idx in range(len(hand_landmarks_list)):
                hand_landmarks = hand_landmarks_list[idx]
                handedness = handedness_list[idx]

                # Draw the hand landmarks.
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
                ])
                solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                solutions.hands.HAND_CONNECTIONS,
                solutions.drawing_styles.get_default_hand_landmarks_style(),
                solutions.drawing_styles.get_default_hand_connections_style())

                # Get the top left corner of the detected hand's bounding box.
                height, width, _ = annotated_image.shape
                x_coordinates = [landmark.x for landmark in hand_landmarks]
                y_coordinates = [landmark.y for landmark in hand_landmarks]
                text_x = int(min(x_coordinates) * width)
                text_y = int(min(y_coordinates) * height) - MARGIN

                # Draw handedness (left or right hand) on the image.
                cv2.putText(annotated_image, f"{handedness[0].category_name}",
                            (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                            FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

            return annotated_image


# ------------------------------------------------------------------------------------------------

def main():
    for file_name in IMAGE_FILES:
        '''
        This is where hand landmarks are calculated
        >>>>
        '''
    
        # Load the input image.
        image = mp.Image.create_from_file(f'input_images/{file_name}')

        # Run the hand tracking model
        detection_result = detector.detect(image)

        # Draw landmarks on our input image
        annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)

        # Save the hand detection result to a json file for later processing
        with open(f'./output_json/{file_name}', 'w') as f:
            f.write(json.dumps(detection_result.__dict__, default=dumper, indent=2))
        
        # cv2.imshow('window', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        
        # Save the labeled image
        cv2.imwrite(f'./labeled_images/{file_name}', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

        
        '''
        <<<<
        This is the end of where hand landmarks are calculated
        '''

        ###################################


        '''
        This is where temporal starts...
        >>>>
        '''
        
        testImage = load_file(file_name)
        
        temporalImageArray = createTemporalImage(testImage)
        
        temporalImageArray = np.array([temporalImageArray])
        temporalImage = Image.fromarray(temporalImageArray, 'RGB')
        temporalImage.save(f'./output_images/{file_name}')

        '''
        <<<<
        This is where temporal ends...
        '''

    displayResults(IMAGE_FILES)


if __name__ == "__main__":
    main()
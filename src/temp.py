import threading
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from TemporalMap import TemporalMap

# Load the pre-trained model
model = tf.keras.models.load_model(r'C:\Users\awebb\Documents\Programming\Python\Unnamed\models\direction_predictor.keras')

# Initialize the TemporalMap object
tmapMaker = TemporalMap(RENDER_TMAPS=False, MODEL_PATH=r'C:\Users\awebb\Documents\Programming\Python\Unnamed\models\hand_landmarker.task')
tmapMaker.setup()

# Constants
img_height = 30
img_width = 42
class_names = ['l2r', 'r2l']

# Open the default camera
cam = cv2.VideoCapture(0)
frame_lock = threading.Lock()
frame = None
running = True


def capture_frames():
    global frame, running
    while running:
        ret, temp_frame = cam.read()
        if ret:
            with frame_lock:
                frame = temp_frame.copy()


def process_and_predict():
    global frame, running
    while running:
        if frame is not None:
            with frame_lock:
                current_frame = frame.copy()

            # Convert frame to mediapipe image
            mpFrame = tmapMaker.convertImageToMediapipeImage(current_frame)

            # Calculate hand landmarks
            handLandmarks = tmapMaker.calculateHandLandmarks(mpFrame)

            # Calculate and append the new Timg (temporal image) to the tmap
            tmapMaker.tmap.append(tmapMaker.calculateTimg(handLandmarks))

            # Convert the tmap (2D array with RGB values) into a format suitable for PIL
            tmap_array = np.array(tmapMaker.tmap)

            # Convert the NumPy array to PIL Image
            img = Image.fromarray(tmap_array.astype('uint8'), 'RGB')
            img = img.resize((img_width, img_height))

            # Convert the resized PIL image to a NumPy array
            img_array = image.img_to_array(img)

            # Add a batch dimension (model expects a batch of images)
            img_array = np.expand_dims(img_array, axis=0)

            # Make a prediction using the model
            predictions = model.predict(img_array, verbose=0)

            # Convert logits to probabilities using softmax
            probabilities = tf.nn.softmax(predictions[0])
            predictedClass = class_names[np.argmax(probabilities)]

            # Print the predicted class
            print(predictedClass)


def display_frames():
    global frame, running
    while running:
        if frame is not None:
            with frame_lock:
                current_frame = frame.copy()

            # Display the captured frame
            cv2.imshow('Camera', current_frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord('q'):
            running = False
            break


# Create and start threads
capture_thread = threading.Thread(target=capture_frames)
process_thread = threading.Thread(target=process_and_predict)
display_thread = threading.Thread(target=display_frames)

capture_thread.start()
process_thread.start()
display_thread.start()

# Wait for threads to finish
capture_thread.join()
process_thread.join()
display_thread.join()

# Release the capture and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()

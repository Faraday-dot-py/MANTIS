import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2  # Import OpenCV for displaying images
from tensorflow.keras.preprocessing import image
from lib.TemporalMap import TemporalMap

# Load the saved model
model = tf.keras.models.load_model('DirectionPredictor.h5')

# Initialize the TemporalMap
tmap = TemporalMap(CONTEXT_WINDOW=30)
tmap.setup()

# Define image preprocessing function
def preprocess_image(img, img_size=(30, 42)):
    # Load and resize the image
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize pixel values
    
    # Add a batch dimension
    img_array = np.expand_dims(img_array, axis=0)  # Shape becomes (1, 30, 42, 3)
    return img_array

# Create a figure and subplots for visualization
fig, ax = plt.subplots(1, 2, figsize=(10, 5))  # Two subplots side by side

while True:
    # Capture the frame from TemporalMap
    frame = tmap.captureImage()
    mpFrame = tmap.convertImageToMediapipeImage(frame)
    handLandmarks = tmap.calculateHandLandmarks(mpFrame)        
    tmap.tmap.append(tmap.calculateTimg(handLandmarks))

    # Preprocess the temporal map
    preprocessedHandMap = preprocess_image(tmap.tmap)

    # Get model prediction
    prediction = model.predict(preprocessedHandMap)
    # print("--> ", prediction[0], type(prediction))

    # Display camera view using OpenCV
    cv2.imshow('Camera View', frame)

    # Plot the temporal map and model predictions using Matplotlib
    ax[0].imshow(tmap.tmap)  # Show the latest temporal map
    ax[1].bar([1, 2], prediction[0])  # Display model outputs as bar chart
    
    plt.draw()
    plt.pause(0.001)

    # Refresh temporal map display
    tmap.refreshTmap()

    # Break loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
tmap.cap.release()

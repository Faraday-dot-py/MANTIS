# Import necessary libraries
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np

# Initialize MediaPipe Holistic model
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()

# Open a webcam
cap = cv2.VideoCapture(0)

# Create a matplotlib figure and axis for plotting
plt.ion()  # Turn on interactive mode for live updates
fig, ax = plt.subplots()
sc = ax.scatter([], [])  # Placeholder scatter plot for landmarks

def update_plot(landmarks):
    ax.clear()  # Clear the plot to update it with new points
    ax.set_xlim(0, 1)  # x-axis limits (MediaPipe gives normalized values)
    ax.set_ylim(0, 1)  # y-axis limits (MediaPipe gives normalized values)

    # Extract (x, y) coordinates from the landmarks
    x_data = [landmark.x for landmark in landmarks]
    y_data = [landmark.y for landmark in landmarks]

    # Plot the new points on the graph
    sc = ax.scatter(x_data, y_data, c='r')  # Scatter plot
    plt.draw()  # Update the figure
    plt.pause(0.001)  # Pause for a brief moment to allow live updates

# Main loop to process webcam frames
while cap.isOpened():
    # Read frames from the webcam
    ret, frame = cap.read()
    if not ret:
        print("frames not found")
        break

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame using the Holistic model
    results = holistic.process(rgb_frame)

    # Check if there are any landmarks detected for the right hand
    if results.right_hand_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        
        # Extract the landmarks for plotting
        right_hand_landmarks = results.right_hand_landmarks.landmark

        # Update the plot with the current hand landmarks
        update_plot(right_hand_landmarks)

    # Display the annotated frame
    cv2.imshow("output", frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
plt.ioff()  # Turn off interactive mode
plt.show()  # Keep the final plot displayed

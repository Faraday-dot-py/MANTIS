from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import os
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

# Open an image file
file_path = r"training_data\unprocessed_training_data\test"
rootPath = os.getcwd()
prevRootPath = str(Path(rootPath).parents[1])
tmapPath = prevRootPath + "\\" + file_path + r"\sample_3.png"
loadedTmapImage = Image.open(tmapPath)

# Convert the image to a numpy array
loadedTmapArray = np.array(loadedTmapImage)

# Horizontal transformation
def horizontal_transform(points, distance):
    return [[x + distance, y] for x, y in points]

# Function to create 3D hand animations
def create_multiple_hand_animations_3d(image_array, interval=1/30):
    """
    Function to create animations based on image arrays and hand connections in 3D.

    Parameters:
    - image_array: numpy array of the image to process.
    - interval: Time interval between frames (in seconds).

    Returns:
    - Displays 3D animation using matplotlib.
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    hand_connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),        # Index Finger
        (0, 9), (9, 10), (10, 11), (11, 12),   # Middle Finger
        (0, 13), (13, 14), (14, 15), (15, 16), # Ring Finger
        (0, 17), (17, 18), (18, 19), (19, 20), # Pinky Finger
        (5, 9), (9, 13), (13, 17)              # Palm
    ]
    
    # Extract the x, y, z coordinates from the image (red, green, blue channels)
    x = image_array[:, :, 0]
    y = image_array[:, :, 1]
    z = image_array[:, :, 2]

    ax.view_init(elev=30, azim=-60)  # Adjust this to set the viewpoint
    
    # Update function for the animation
    def update(num):
        ax.cla()  # Clear the axis
        ax.set_xlim(0, 255)
        ax.set_ylim(0, 255)
        ax.set_zlim(255,0)
        ax.set_title(f'Frame {num}')
        
        # Scatter plot for the points (landmarks)
        # Swap y and z here: use x for horizontal, y for vertical, and z for depth
        ax.scatter(x[num], z[num], y[num], c='red')  # Plot landmarks in red

        # Draw the connections between landmarks
        for connection in hand_connections:
            start_idx, end_idx = connection
            ax.plot([x[num][start_idx], x[num][end_idx]], 
                    [z[num][start_idx], z[num][end_idx]], 
                    [y[num][start_idx], y[num][end_idx]], 
                    'blue')  # Draw connections in blue
            
        ax.view_init(elev=30, azim=num)  # Rotate the view over time

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=list(range(len(x) - 1)), interval=interval)

    # Display the animation
    plt.show()

# Apply the 3D animation function to the loaded image array
create_multiple_hand_animations_3d(loadedTmapArray)


from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import cv2
from tqdm import tqdm


# Open an image file (we will overlay points on this video, not the image)
tmapPath = r'C:\Users\awebb\Documents\Programming\Python\MANTIS\training_data\unprocessed_training_data\hello\sample_1.png'
loadedTmapImage = Image.open(tmapPath)

# Convert the image to a numpy array (not directly used in the video but kept)
loadedTmapArray = np.array(loadedTmapImage)


# Horizontal transformation
def horizontal_transform(points, distance):
    return [[x + distance, y] for x, y in points]

# Vertical transformation
def vertical_transform(points, distance):
    # FILL ME IN
    return [[x, y + distance] for x, y in points]

# Rotational transformation
def rotate_points(points, angleDegrees, centerPointIndex):
    """
    Rotate a set of points around a specified center point.
    
    Parameters:
    - points: List of [x, y] coordinates.
    - angle_degrees: The angle to rotate the points (in degrees).
    - center_point: The point [cx, cy] around which to rotate.
    
    Returns:
    - List of rotated points.
    """
    # FILL ME IN
    angleRadians = np.radians(angleDegrees)
    # cx, cy = points[centerPointIndex]  # Center of rotation
    cx, cy = 255*0.5, 255*0.5

    # Perform the rotation
    rotated_points = []
    for i, (x, y) in enumerate(points):
        # if i < len(points)/2:
        # cx, cy = points[centerPointIndex] 
        # else:
        #     cx, cy = points[centerPointIndex]
        new_x = cx + np.cos(angleRadians) * (x - cx) - np.sin(angleRadians) * (y - cy)
        new_y = cy + np.sin(angleRadians) * (x - cx) + np.cos(angleRadians) * (y - cy)
        rotated_points.append([new_x, new_y])

    return rotated_points


# Scale transformation
def scale_points(points, scale):
    """
    Scale a set of points around the origin.
    
    Parameters:
    - points: List of [x, y] coordinates.
    - scale: The scale factor.
    
    Returns:
    - List of scaled points.
    """
    # FILL ME IN
    return [[x * scale, y * scale] for x, y in points]

# Shear transformation
def shear_points(points, shearX, shearY):
    """
    Shear a set of points.
    
    Parameters:
    - points: List of [x, y] coordinates.
    - shearX: The shear factor in the x direction.
    - shearY: The shear factor in the y direction.
    
    Returns:
    - List of sheared points.
    """
    # FILL ME IN
    sheared_points = []
    for x, y in points:
        new_x = x + shearX * y
        new_y = y + shearY * x
        sheared_points.append([new_x, new_y])
    
    return sheared_points


def create_multiple_hand_animations(image_arrays, interval=1/30):
    """
    Function to create multiple animations based on a list of image arrays and hand connections.

    Parameters:
    - image_arrays: List of numpy arrays of the images to process.
    - interval: Time interval between frames (in seconds).

    Returns:
    - None. Displays multiple animations using matplotlib.
    """

    # Set up the figure and axes for each image_array
    num_images = len(image_arrays)
    fig, axes = plt.subplots(1, num_images, figsize=(5*num_images, 5))

    # Preprocess the image arrays to extract the x and y coordinates for each one
    hand_connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),        # Index Finger
        (0, 9), (9, 10), (10, 11), (11, 12),   # Middle Finger
        (0, 13), (13, 14), (14, 15), (15, 16), # Ring Finger
        (0, 17), (17, 18), (18, 19), (19, 20), # Pinky Finger
        (5, 9), (9, 13), (13, 17)              # Palm
    ]

    data = []
    for image_array in image_arrays:
        half = int(len(image_array[0]) / 2) * 0
        x = image_array[:, :, 0]
        y = image_array[:, :, 1]
        data.append((x, y))  # Append each x, y pair to data

    # Update function for the animation
    def update(num):
        for i, ax in enumerate(axes):
            ax.cla()  # Clear the axis
            ax.set_xlim(0, 255)
            ax.set_ylim(0, 255)
            ax.set_title(f'Frame {num}')
            
            x, y = data[i]
            
            # Scatter plot for the points (landmarks)
            ax.scatter(x[num], y[num], c='red')  # Plot the landmarks in red

            # Draw the connections between landmarks
            for connection in hand_connections:
                # Get the coordinates of the connected points
                start_idx, end_idx = connection
                ax.plot([x[num][start_idx], x[num][end_idx]], 
                        [y[num][start_idx], y[num][end_idx]], 
                        'blue')  # Draw the connections in blue

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=(list(range(len(data[0][0]) - 1))), interval=interval)

    # Display the animation
    plt.show()


# Apply all transformations to the points, creating a new set of points for each transformation
loadedTmapWithoutBlue = np.array([[(landmark[0], landmark[1]) for landmark in frame] for frame in loadedTmapArray])

# Apply horizontal transformation to each frame of the tmap
# horizontalTmap = np.array([horizontal_transform(frame, 50) for frame in loadedTmap])  # Move right by 50 units

# # Apply vertical transformation to each frame of the tmap
# verticalTmap = np.array([vertical_transform(frame, 50) for frame in loadedTmap])  # Move up by 50 units

# # Apply rotational transformation to each frame of the tmap
# rotatedTmap = np.array([rotate_points(frame, 45, 9) for frame in loadedTmap])  # Rotate by 45 degrees around point 0

# # Apply scaling transformation to each frame of the tmap
# scaledTmap = np.array([scale_points(frame, 0.5) for frame in loadedTmap])  # Scale down by a factor of 0.5

# # Apply shearing transformation to each frame of the tmap
# shearedTmap = np.array([shear_points(frame, 0.1, 0.1) for frame in loadedTmap])  # Shear by 0.5 in both directions

xRange = range(-10, 10)
yRange = range(-10, 10)
rotationRange = range(-12, 12)
scaleRange = np.linspace(0.1, 2, 10)
shearRange = [(x, y) for x in np.linspace(-0.2, 0.2, 10) for y in np.linspace(-0.2, 0.2, 10)]

generatedTmaps = []

print("Generating Tmaps...")
print("Horizontal Transformations")
# generate and save every possible tmap with the ranges provided
for x in xRange:
    horizontalTmap = np.array([horizontal_transform(frame, x) for frame in loadedTmapWithoutBlue])
    generatedTmaps.append(horizontalTmap)

print("Vertical Transformations")
for y in yRange:
    verticalTmap = np.array([vertical_transform(frame, y) for frame in loadedTmapWithoutBlue])
    generatedTmaps.append(verticalTmap)

print("Rotational Transformations")
for angle in rotationRange:
    rotatedTmap = np.array([rotate_points(frame, angle, 9) for frame in loadedTmapWithoutBlue])
    generatedTmaps.append(rotatedTmap)

print("Scaling Transformations")
for scale in scaleRange:
    scaledTmap = np.array([scale_points(frame, scale) for frame in loadedTmapWithoutBlue])
    generatedTmaps.append(scaledTmap)

print("Shearing Transformations")
for shear in shearRange:
    shearedTmap = np.array([shear_points(frame, shear[0], shear[1]) for frame in loadedTmapWithoutBlue])
    generatedTmaps.append(shearedTmap)

def restore_blue_channel(generated_tmap, original_tmap):
    """
    Restore the blue color channel from the original tmap back into the generated tmap.

    Parameters:
    - generated_tmap: A numpy array representing the generated image with the blue channel removed.
    - original_tmap: A numpy array representing the original image.

    Returns:
    - Modified generated tmap with the blue channel restored from the original tmap.
    """
    # Copy the blue channel (3rd channel, index 2) from the original tmap to the generated tmap
    restored_tmap = np.copy(generated_tmap)
    blue_channel = original_tmap[:, :, 2]
    restored_tmap = np.dstack((restored_tmap, blue_channel))
    return restored_tmap

def restore_blue_channel(generated_tmap, original_tmap):
    
    # Copy the generated tmap to avoid modifying the original data
    restored_tmap = np.copy(generated_tmap)
    
    # Restore the blue channel (channel index 2 in RGB)
    new_channel = original_tmap[:, :, 2]
    restored_tmap = np.dstack((restored_tmap, new_channel))
    
    return restored_tmap


# Save all the tmaps to a folder with a progress bar
total_frames = sum(len(tmap) for tmap in generatedTmaps)
with tqdm(total=total_frames, desc="Saving Tmaps") as pbar:
    for i, tmap in enumerate(generatedTmaps):
        fullTmap = restore_blue_channel(tmap, loadedTmapArray)
        fullTmapImage = Image.fromarray(fullTmap.astype('uint8'))
        fullTmapImage.save(fr"C:\Users\awebb\Documents\Programming\Python\MANTIS\src\TmapVis\generatedTmaps\generated_tmap_{i}.png")

        pbar.update(len(loadedTmapWithoutBlue))

# tmaps = np.array([
#     loadedTmap,
#     # horizontalTmap,
#     # verticalTmap,
#     rotatedTmap,
#     # scaledTmap,
#     # shearedTmap
# ])

# create_multiple_hand_animations(tmaps)
# # print(np.array(horizontalTmap).shape)



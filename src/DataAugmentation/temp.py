# %%
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import cv2

# %%
# Open an image file (we will overlay points on this video, not the image)
tmapPath = r'C:\Users\awebb\Documents\Programming\Python\MANTIS\training_data\unprocessed_training_data\hello\sample_1.png'
tmapImage = Image.open(tmapPath)

# Convert the image to a numpy array (not directly used in the video but kept)
tmap = np.array(tmapImage)

# %%
# Horizontal transformation
def horizontal_transform(x, y, width, height):
    return x, height - y

# %%
# Vertical transformation
def vertical_transform(x, y, width, height):
    return width - x, y

# %%
# Rotational transformation
import numpy as np

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
    # Convert the angle to radians
    angle_radians = np.radians(angleDegrees)
    
    # Get the center point
    centerPoint = points[centerPointIndex]

    # Create the rotation matrix
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians)],
        [np.sin(angle_radians), np.cos(angle_radians)]
    ])
    
    # Translate points to rotate around the origin
    translated_points = np.array(points) - centerPoint
    
    # Apply the rotation
    rotated_points = np.dot(translated_points, rotation_matrix)
    
    # Translate points back to the original position
    final_points = rotated_points + centerPoint
    
    return final_points.tolist()



# %%
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
    # Create the scale matrix
    scale_matrix = np.array([
        [scale, 0],
        [0, scale]
    ])
    
    # Apply the scale
    scaled_points = np.dot(points, scale_matrix)
    
    return scaled_points.tolist()

# %%
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
    # Create the shear matrix
    shear_matrix = np.array([
        [1, shearX],
        [shearY, 1]
    ])
    
    # Apply the shear
    sheared_points = np.dot(points, shear_matrix)
    
    return sheared_points.tolist()

# %%
# Apply all transformations to the points, creating a new set of points for each transformation
loadedTmap = [[(landmark[0], landmark[1]) for landmark in frame] for frame in tmap]

# Apply horizontal transformation to each frame of the tmap
horizontalTmap = [[horizontal_transform(landmark[0], landmark[1], tmapImage.width, tmapImage.height) for landmark in frame] for frame in loadedTmap]

plt.imshow(loadedTmap)


# # %%
# # Display an animation of the original tmap and the transformed points
# fig, ax = plt.subplots()
# ax.set_aspect('equal')
# ax.set_title('Tmap')

# # Create a scatter plot of the original tmap
# scat = ax.scatter([], [], s=10, color='blue')
# scat.set_offsets(loadedTmap[0])

# # Create a scatter plot of the transformed tmap
# scat2 = ax.scatter([], [], s=10, color='red')
# scat2.set_offsets(horizontalTmap[0])

# # Update function for the animation
# def update(frame):
#     # print(frame)
#     scat.set_offsets(loadedTmap[frame])
#     scat2.set_offsets(horizontalTmap[frame])
#     return scat, scat2

# # Create the animation
# ani = animation.FuncAnimation(fig, update, frames=len(loadedTmap), interval=100, blit=True)
# plt.show()





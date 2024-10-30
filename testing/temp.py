import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Sample landmark connections for drawing hand skeleton
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),    # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),    # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12), # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
]

# Function to load JSON data from a file
def load_movement_data(json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    return data['hand_landmarks']

# Example: Load movement data from a JSON file
json_file_path = r'C:\Users\awebb\Documents\Programming\Python\MANTIS\project_documentation\triple_comparison_test\jacob_1.json'  # Path to your JSON file
hand_landmarks = load_movement_data(json_file_path)

# Convert landmark data to a format usable for plotting
def extract_landmarks(frame):
    x = [point['x'] for point in frame]
    y = [point['y'] for point in frame]
    return np.array(x), np.array(y)

# Initialize the plot
fig, ax = plt.subplots()
ax.set_xlim(0, 1)  # Assuming normalized coordinates (0 to 1)
ax.set_ylim(0, 1)
points, = ax.plot([], [], 'bo')  # Blue points for landmarks
lines = [ax.plot([], [], 'k-')[0] for _ in connections]  # Lines for hand skeleton

# Initialize empty points and lines
def init():
    points.set_data([], [])
    for line in lines:
        line.set_data([], [])
    return points, *lines

# Update function for each frame
def update(frame_num):
    x, y = extract_landmarks(hand_landmarks[frame_num])
    points.set_data(x, y)

    # Update lines connecting landmarks
    for idx, (start, end) in enumerate(connections):
        lines[idx].set_data([x[start], x[end]], [y[start], y[end]])

    print("ran")

    return points, *lines
print(len(hand_landmarks))
# Create the animation
ani = FuncAnimation(fig, update, frames=len(hand_landmarks),
                    init_func=init, blit=True, interval=100)  # Adjust interval for speed

plt.show()

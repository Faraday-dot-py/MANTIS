# Research Objective
Develop a novel approach detect patterns in the motion of a series of 3D cartesian coordinate points. This methodology will be tested by translating American Sign Language gestures to English.

# Methodology

### Data Representation
A set of 3-Dimensional cartesian coordinates can be represented with the notation $$(x, y, z)$$with each variable representing the position of a point along its respective axis. To represent the movement of a point over time, we can create a list of these points with equal timesteps in between each point:$$[(x_{t=1}, y_{t=1}, z_{t=1}), (x_{t=2}, y_{t=2}, _{t=2}),...]$$Similarly, to represent a set of points moving through time, we can create a 2-dimensional list consisting of these movement lists:
$$\left[ \begin{array}{l} [(x^{p=1}_{t=1}, y^{p=1}_{t=1}, z^{p=1}_{t=1}), (x^{p=1}_{t=2}, y^{p=1}_{t=2}, z^{p=1}_{t=2}), \cdots], \\ [(x^{p=2}_{t=1}, y^{p=2}_{t=1}, z^{p=2}_{t=1}), (x^{p=2}_{t=2}, y^{p=2}_{t=2}, z^{p=2}_{t=2}), \cdots], \\ \cdots \\ \end{array} \right]$$

Similarly, an 8-bit color can be represented with the notation $(r, g, b)$, with each variable representing the red, green, and blue brightness values. 2D images follow a similar format to the earlier mentioned points moving through time. The following algorithm can be used to convert a set of Mediapipe hand landmarks to a ``temporal image``. A temporal image represents the positions of a set of points in an instant in time, and can be visualized by displaying the array as a line of pixels. These temporal images can be layered together in order of age to form a ``temporal map``, or tmap, which represents the movement of a set of points over time. Below is one such example image.
![[Demo Temporal Map.png]]

This Tmap can then be piped into a basic image classification model to classify the motion of the points and extract information from a relatively simple image.
### Applying Machine Learning
Coincidentally, one method of creating images is using a data format similar to the structure described above. A 2-dimensional array of 3d vectors represents a 2d set of RGB pixels that can be displayed on a screen or analyzed by various image editing tools.

Many neural networks have been designed specifically for the analysis and manipulation of RGB images. Image classification models, models that label images based on their content, have been used for decades, starting with [HERE]. Modern versions of these models apply various algorithms to detect small patterns in images, and patterns in those patterns, etc, to classify an image based on its content. A similar approach is being applied here to detect patterns in a Tmap, giving us the capability to detect patterns in motion data with models proven to be efficient and effective.
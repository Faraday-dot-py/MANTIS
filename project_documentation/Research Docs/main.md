# Research Objective
Develop a novel approach detect patterns in the motion of a series of 3D cartesian coordinate points. This methodology will be tested by translating American Sign Language gestures to English.

# Methodology

### Data Representation
A set of 3-Dimensional cartesian coordinates can be represented with the notation $$(x, y, z)$$with each variable representing the position of a point along its respective axis. To represent the movement of a point over time, we can create a list of these points with equal timesteps in between each point:$$[(x_{t=1}, y_{t=1}, z_{t=1}), (x_{t=2}, y_{t=2}, _{t=2}),...]$$Similarly, to represent a set of points moving through time, we can create a 2-dimensional list consisting of these movement lists:
$$\left[ \begin{array}{l} [(x^{p=1}_{t=1}, y^{p=1}_{t=1}, z^{p=1}_{t=1}), (x^{p=1}_{t=2}, y^{p=1}_{t=2}, z^{p=1}_{t=2}), \cdots], \\ [(x^{p=2}_{t=1}, y^{p=2}_{t=1}, z^{p=2}_{t=1}), (x^{p=2}_{t=2}, y^{p=2}_{t=2}, z^{p=2}_{t=2}), \cdots], \\ \cdots \\ \end{array} \right]$$

Similarly, an 8-bit color can be represented with the notation $(r, g, b)$, with each variable representing the red, green, and blue brightness values. 2D images follow a similar format to the earlier mentioned points moving through time. The following algorithm can be used to convert a set of Mediapipe hand landmarks to a ``temporal image``. A temporal image represents the positions of a set of points in an instant in time, and can be visualized by displaying the array as a line of pixels. These temporal images can be layered together in order of age to form a ``temporal map``, which represents the movement of a set of points over time. Below is one such example image.
![[Pasted image 20240911155038.png]]

This temporal map can then be piped into a basic image classification model to classify the motion of the points and extract information from a relatively simple image.
### Applying Machine Learning

A basic image re
Average of 10 Trials:

| Model Number | Trainable Params | Training Time | Accuracy | Validation Accuracy | Loss   | Validation Loss |
| ------------ | ---------------- | ------------- | -------- | ------------------- | ------ | --------------- |
| 1            | 2398             | 2.15969       | 0.5623   | 0.445               | 0.6842 | 0.6993          |
| 2            | 9354             | 2.31966       | 0.6134   | 0.53                | 0.6651 | 0.6802          |
| 3            | 20870            | 2.37605       | 0.6646   | 0.57                | 0.6463 | 0.6673          |
| 4            | 36946            | 2.14701       | 0.7597   | 0.695               | 0.6049 | 0.6258          |
| 5            | 57582            | 2.25689       | 0.711    | 0.56                | 0.595  | 0.6281          |
| 6            | 82778            | 2.29627       | 0.8161   | 0.825               | 0.5645 | 0.5893          |
| 7            | 112534           | 2.34378       | 0.8233   | 0.885               | 0.5308 | 0.5615          |
| 8            | 146850           | 2.11332       | 0.8525   | 0.89                | 0.5274 | 0.546           |
| 9            | 185726           | 2.12047       | 0.906    | 0.96                | 0.4699 | 0.4823          |
| 10           | 229162           | 2.19771       | 0.9647   | 0.995               | 0.4288 | 0.4334          |
| 11           | 277158           | 1.87676       | 0.9537   | 0.995               | 0.448  | 0.4428          |
| 12           | 329714           | 1.9211        | 0.983    | 1                   | 0.3704 | 0.365           |
| 13           | 386830           | 1.97393       | 0.9817   | 1                   | 0.3681 | 0.3578          |
| 14           | 448506           | 1.98703       | 0.9976   | 1                   | 0.2905 | 0.2737          |
| 15           | 514742           | 2.03616       | 0.9976   | 1                   | 0.2842 | 0.2653          |


*Tests performed on 13th gen Intel(R) i7-1360P*


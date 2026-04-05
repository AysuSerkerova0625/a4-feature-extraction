# Edge Analysis and Interest Point Detection

In this task I've analyzed an image to find important features like edges, corners, shapes, and key points. The code detects edges to show object boundaries, found corners where the image changes sharply, and uses interest points to match the same object from different images. These steps help understand the structure of the image and identify objects.

# How to Run
```
git clone <repo>
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


# Overview
The tasks include:

- Edge detection
- Corner detection
- Line and circle detection
- Active contour (snakes)
- Interest point detection and matching using ORB

The input images were taken using a mobile phone. The selected object contains visible lines and circular shapes, making it suitable for edge and shape analysis.

### edgeCornerShape.py
```
python3 src/edgeCornerShape.py
```

This code is used to detect edges, corners, lines, and circles. Firstly, the noise from the image is removed using blur, then Canny detection algorihtm is used for edges, Harris for corners, and Hough Transform to detect lines and circles.
As an additional note, the Harris corner detector did not perform very well on this image. This method works better on images with clearer structure, smoother edges, and higher visual quality. Since my image contains some irregularities, the detected corners are not as accurate as expected.

### contour.py
```
python3 src/contour.py
```

In this file, an interactive Active Contour (snake) algorithm is applied to detect the boundary of an object. The user can click several points around the object to give an initial rough outline. Then, the algorithm will automatically move and adjuste this outline so that it fits the actual edges of the object more accurately. 

### orb.py
```
python3 src/orb.py
```
ORB is implemented to detect keypoints and match the same object in two different images. It finds important points and compares them to identify similarities between images taken from different perspectives.

# Conclusion
Overall, this project shows how different computer vision techniques can be used together to analyze an image. Each method highlights a different type of feature, such as edges, corners, shapes, or keypoints. The results demonstrate how images can be better understood and how the same object can be detected even from different perspectives.
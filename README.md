# lane-detection
Implementation of lane detection and turn prediction with OpenCV based on Python.

The steps to successfully implement the whole process are as follows: 

* Undistort each frame with the given camera calibration matrix and distortion coefficients.
* Implement color segmentation in different color space and gradients to acquire a binary image.
* Apply perspective transformation(in this case, Homography) to get the top-down view(bird view).
* Based on the histogram of each frame, apply sliding windows to find the boundaries of each side of the lane.
* Determine the curvature of the lane as well as vehicle position with respect to center of the lane.
* Project the mask back to the original image.
* Visualize the output and predict the turn based on the curvature computed previously. 

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

## Pipeline demostration ##
### 1.Undistortion ###
#### Original image: ####
![](/images/images/original.png)

#### Undistorted image: ####
![](/images/images/ori_undist.png)

### 2.Thresholding ###
To maximally extract the line of the lane, I applied thresholding on two color spaces, which are HSV and HLS respectively. Also to include the missing part from the thresholding, I implemented Sobel on the image as well to get the gradients. The combined binary image is as follows: 

![](/images/images/threshold.png)

### 3.Homography ###
![](/images/images/warpped.png)
As you can see the projected image looks a bit larger compared to the original image. That is because the original projection may miss the part of the edges when the vehicle is making turns.

![](/images/images/warpped_cut.png)

### 4.Sliding_window ###
![](/images/images/sliding_window.png)

### 5.Draw polyline ###
![](/images/images/poly.png)

### 6.Back-projection ###
![](/images/images/final.png)

### 7.Turn prediction ###
![](/images/images/turn_predict.png)

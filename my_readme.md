## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./output_images/thresholded_gradient.png "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./output_images/polyfit.png "Fit Visual"
[image6]: ./output_images/final_result.png "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README
  
### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second code cell of the IPython notebook located in "./examples/example.ipynb"

I start by preparing object points, which are the coordinates of the corners of the chessboard. Then i try to find the corresponding corners in the image with the opencv function 'findChessboardCorners' and store the coordinates in a list called imgpoints.
When every corner is found i use the opencv functions 'calibrateCamera' and 'undistort' to get the camera image and then undistort the image with this matrix. 

# insert chessboard images
![alt text][image1]


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image, which can be found in the 4th Code cell of the IPython notebook

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp_image()`, which appears in the 8th code cell of the IPython notebook.  The `warp_image()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose to hardcode the source and destination points in the following manner:

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 568, 470      | 200, 0        | 
| 260, 680      | 200, 680      |
| 1043, 680     | 1000, 680     |
| 717, 470      | 1000, 0       |


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I am identifying the lane lines by taking the histogram of the bottom half columns of the image. There you should expect to see two peeks for left and right lane, which gives you the location in the image. Combined with a sliding window you can get a pretty good idea about where the lines are in your image. After determining the location i am fitting a 2nd order polynomial around the lane lines (code cell #10), which results in an image like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius curvature is calculated in code cell # 12. The pixel values of the lane are scaled into meters using the scaling factors defined as follows:

xm_per_pix = 3.7/700  # meters per pixel in x

ym_per_pix = 30/720   # meters per pixel in y

These values are then used to compute the polynomial coefficients in meters and then the formula given in the class is used to calculate the radius of curvature.

The position of the vehicle is calculated by assuming the camera to be centered in the vehicle and checking the distance of the midpoint of the two lanes from the center of the image.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.
The last task was to warp the filled polynomial back to the image perspective. I implemented this in code cell #14 in the function called 'draw_final_result'. Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The approach i took is to take a part of the image and warp it to a top down perspective. On the resulting image i take a histogram of the colors to find the lane markers and try to fit a polynomial through the resulting points.

The approach works reasonably well on the provided video, but but most definitely fail if there were no clear enough lane markers in the warped part of the image. To get past that i would need to either expand the image area to look for lanes, or implement an average over multiple images.

Another scenario which would cause the approach to fail would be any other marker on the road. For example a temporary lane in a construction area. The histogram approach would then indicate more than two locations, which would cause the approach to fail.

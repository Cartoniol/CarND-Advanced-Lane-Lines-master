# Advanced Lane Finding SW Writeup

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.

![alt text][image1]

* Apply a distortion correction to raw images.

![alt text][image2]

* Apply a perspective transform to rectify binary image ("birds-eye view").

![alt text][image4]

* Use color transforms, gradients, etc., to create a thresholded binary image.

![alt text][image3]

* Detect lane pixels and fit to find the lane boundary.

![alt text][image5]

* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

![alt text][image6]

The code is stored in the file [P2.ipynb](./P2.ipynb).

[//]: # "Image References"

[image1]: ./output_images/ChessUndist.png "Chessboard Undistorted"
[Image2]: ./output_images/undist_test4.jpg "Undistorted"
[image3]: ./output_images/combined3_test4.jpg "Binary Warped Example"
[image4]: ./output_images/warped_test4.jpg "Warp Example"
[image5]: ./output_images/combined_test4.jpg "Fit Visual"
[image6]: ./output_images/result_test4.jpg "Output"
[video1]: ./output_videos/project_video.mp4 "Video"

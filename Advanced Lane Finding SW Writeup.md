# Advanced Lane Finding SW Writeup

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

[//]: # "Image References"

[image1]: ./output_images/undist_test4.jpg "Undistorted"
[image2]: ./test_images/test4.jpg "Road Transformed"
[image3]: ./output_images/combined3_test4.jpg "Binary Warped Example"
[image4]: ./output_images/warped_test4.jpg "Warp Example"
[image5]: ./output_images/combined_test4.jpg "Fit Visual"
[image6]: ./output_images/result_test4.jpg "Output"
[image7]: ./output_images/ChessUndist.png "Chessboard Undistorted"
[video1]: ./output_videos/project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### In the following, the main functions used for the lane recognition code are described together with the rubric points.

### The rubric points are the main features that the SW needs to have.  Each point has been addressed and its implementation is supported by example images.

---

## Camera Calibration

The code for this step is contained in the second code cell of the IPython notebook located in "./P2.ipynb".

The aim of the code in the 2nd code cell is to process a set of chessboard images taken from different angulation, distance and camera effect, to obtain camera calibration matrix `mtx`and distortion coefficients `dist` via `cv2.calibrateCamera()`, to finally calibrate the `undistortion()` function. The code follows the steps below:

1. `objp` is defined as set of (x, y, z) coordinates of a  chessboard corners in the world. It is assumed that the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image;
2. Iteratively thoughout the calibration images, `cv2.findChessboardCorners()` computes corners from a gray scale verison of the calibration picture and the number of corners to be recognised;
3. If corners coordinates are returned, these coordinates will be appended to `imgpoints` as will be a copy of  `objp`  to `objpoints`;
4. `cv2.calibrateCamera()` then will output `mtx`and `dist`based on the set of coordinates fed into. 
5. `mtx`and `dist` are eventually stored in `cache_cam_cal`

The following function has been defined to be easily called for the distortion correction in the test image and video pipelines.

```
def undistortion(img):
    
    dst = cv2.undistort(img, cache_cam_cal["mtx"], cache_cam_cal["dist"], None, cache_cam_cal["mtx"])
    
    return dst
```

An example of a distorted and undistorted chessboard is shown below:

![alt text][image7]

## Function definition

To keep the structure of the pipeline for the test image and video lane recognition tidy and clear as much as possible, the code has been unpacked in functions and subfunctions, that will be called by each pipeline. Test image pipeline is different form the video processing one, since some functions are not required for the first case.

### - Thresholding and Combination

- In the 3rd code cell, from line 1 to line 191, color space conversion functions (CSCfun) are listed. From Sobel conversion to LAB color space, each function inputs are the raw picture, a set of thresholds and other information required from the specific funtion. They convert the color space and apply the thresholds so to output a binary image.

- At line 191, 3rd code cell, function `combine()` is used to combine the binary images resulting from each CSCfun activated by the user. As input, `combine()` needs a list of binary images, a list of strings (`algorithm`) associated to the CSCfun requested, and two more parameters for the logic operations. It returns the final binary output. 

* At line 241, 3rd code cell, function `threshold()` receives the image, that is going to be passed through the previously described functions, the `algorithm` name list, and the set of parameters related. The funtion then calls the subfunctions listed in `algorithm` and required by the user for the task, and pass through raw image and parameters. It then calls `combine()` and returns `combine()` output.

### - Warping, Sliding Windows and Search Around and Fit Poly

- At line 1, 4th code cell, `warping()` utileses `cv2.getPerspectiveTransform()` and `cv2.warpPerspective` to return the warped image (Perspective Transform) or reverse - depending on the `direction` parameter. `vertices` as input and `dst` are the x and y coordinates of source and destination for the `cv2.getPerspectiveTransform()` function. 

- At line 25, 4th code cell, `find_lane_pixels()` receives the binary warped picture, that is the image after `warping()` and `threshold()` processing, to return the polynomial coefficients fitted to the couples of pixel coordinates identified as lines. The steps that the algorithm follows are depicted below:

  1. apply histogram algorithm to `binary warped` in order to find the lines starting point at coordinate ymax (see code below);

     ```python
     histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
     midpoint = np.int(histogram.shape[0]//2)
     leftx_base = np.argmax(histogram[:midpoint])
     rightx_base = np.argmax(histogram[midpoint:]) + midpoint
     ```

  2. define windows parameters (`nwindows`, `margin`, `minpix`, `window_height`);

  3. identify x and y coordinates of `nonzero` elements in the binary image;

  4. set starting points of left and right windows;

     ```python
     leftx_current = leftx_base
     rightx_current = rightx_base
     ```

  5. while iterate throught windows, first identify nonzero pixel x & y coordinates inside the windows:

     ```python
     good_left_inds = ((nonzeroy >= win_y_low) & 
                       (nonzeroy < win_y_high) & 
                       (nonzerox >= win_xleft_low) & 
                       (nonzerox < win_xleft_high)).nonzero()[0]
     good_right_inds = ((nonzeroy >= win_y_low) & 
                        (nonzeroy < win_y_high) & 
                        (nonzerox >= win_xright_low) & 
                        (nonzerox < win_xright_high)).nonzero()[0]
     ```

  6. Then append results in `left_lane_inds`and `right_lane_inds`;

  7. adjust next windows position based on the minimum number of pixels identified in the previous one:

     ```python
     if len(good_left_inds) > minpix:
       leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
     if len(good_right_inds) > minpix:
       rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
     ```

  8. call `fit_poly()` to fit polynomials to the set of left and right coordinates.

- At line 113, 4th code cell, `search_around_poly()` has same target of `find_lane_pixels()`, but it skips the windows and searches for pixels in the geometric area defined by `left_fit_prev`and `right_fit_prev`. This function is only used in the pipeline for test videos. In fact, `left_fit_prev`and `right_fit_prev` are the polynomial coefficients for left and right fit of previously recognised lines (previous analysed video frame). The steps are the following:

  1. indentify x and y coordinates of nonzero elements in the binary image;

  2. define searching area (`margin`) based on previous identified lines:

     ```python
     left_lane_inds = ((nonzerox > (left_fit_prev[0]*(nonzeroy**2) + left_fit_prev[1]*nonzeroy + left_fit_prev[2] - margin)) & 
                       (nonzerox < (left_fit_prev[0]*(nonzeroy**2) + left_fit_prev[1]*nonzeroy + left_fit_prev[2] + margin)))
     right_lane_inds = ((nonzerox > (right_fit_prev[0]*(nonzeroy**2) + right_fit_prev[1]*nonzeroy + right_fit_prev[2] - margin)) & 
                        (nonzerox < (right_fit_prev[0]*(nonzeroy**2) + right_fit_prev[1]*nonzeroy + right_fit_prev[2] + margin)))
     ```

  3. identify nonzero pixels (x & y coordinates) inside defined searching area (`leftx`, `lefty`, `rightx`, `righty`);

  4. call `fit_poly()` to fit polynomials to the set of left and right coordinates.

- At line 142, 4th code cell, `fit_poly()` fits a second order polynomial to the points identified (`leftx`, `lefty`, `rightx`, `righty`), generating the polynomial coefficients of the lines. 

  ```
  left_fit = np.polyfit(lefty, leftx, deg = 2)
  right_fit = np.polyfit(righty, rightx, deg = 2)
  ```

  The function set a flag "detected" for left and right lines in case `left_fit` and  `right_fit` are not empty. Eventually it calculates `lane_pos` - lane mid-point (at y_max) - and lane horizontal width averaged over y_max, y_max/2 and y_min coordinates:

  ```
  horiz_width_avg = (horiz_width_ymax + horiz_width_ymed + horiz_width_ymin)/3
  lane_pos = (horiz_width_ymax)/2 + left_xpos_ymax
  ```

`fit_poly()` - as`search_around_poly()` and `find_lane_pixels()` - outputs are: 

- `leftx`, `lefty`, `rightx`, `righty` are the pixel coordinates of left and right lines recognised in the warped image;
- `left_fitx`, `right_fitx`, `ploty` are the points belonging to the left and right fitting lines;
- `left_fit`: left line fitting polynomial coefficients;
-  `right_fit`: right line fitting polynomial coefficients;
-  `lane_pos`: lane position in the warped image;
-  `horiz_width_avg`: lane horizontal width averaged over horizontal width at ymax (image bottom), ymax/2 (image y midpoint), ymin (image top);
-  `detected_lines`: list of booleans that state if left or right line were detected by the algorithm.

### - Pixel to M and Curvature Measurement

* In the 5th code cell, `pixel_to_m()` function is used to convert pixels to meters. It outputs the converted polynomial coefficients and offset of the car with respect to the lane center position. [For more information regarding offset see below]

* `measure_curvature_pixels()` calculates the curvature radius of each identified line based on:
  $$
  R_{curve} = [1+(dx/dy)^2]^{3/2}/|d^2x/dy^2|
  $$
  the lines are discribed by: 
  $$
  f(x) = ax^2 + bx + c
  $$
  where a, b and c are the coefficients from `left_fit`and `right_fit`. The derivatives are then calculated as:
  $$
  f'(y) = dx/dy = 2Ay + B
  $$

  $$
  f''(y) = 2A
  $$

  therefore, the left and right curvature radius are:
  $$
  R_{curve} = (1+(2Ay+B)^2)^{3/2}/|2A|
  $$
  The code implemented is as follow:

  ```
  left_curverad = (1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**(3/2)/(np.absolute(2*left_fit[0]))
  right_curverad = (1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**(3/2)/(np.absolute(2*right_fit[0]))
  ```

### - Useful Class definition

In the 6th code cell:

* A Line class has been defined in order to track lines information.
* counterL & counterR class defined for count and reset function.

## Pipeline (single images)

The pipeline for single test images is located in the 7th code cell. In 8th code cell, instead, is stored the code for test images processing. Iterating throughout images, at line 13, 8th code cell, `process_img()` is called.

1. ### Image Distortion Correction

   At line 7, 7th code cell, `undist` is the undistorted image resulting from `undistortion()`. An example is shown below:

   

   ![alt text][image1]

   

2. ### Perspective Transform

   At line 23, 7th code cell, following the distortion correction, to the undistorted picture is subsequently applied the perspective transformation `warping()`, after the definition of `vertices`coordinates. 

   For this implementation, source and destination coordinates where chosen as follow:

   ```
   src = np.float32(
   		[[140, image.shape[0]],
       [image.shape[1]/2-70,image.shape[0]/2+100],
       [image.shape[1]/2+77,image.shape[0]/2+100],
       [image.shape[1]-80,image.shape[0]]])
   dst = np.float32(
       [[150, image.shape[0]],
       [150,0], [image.shape[1]-150,0],
       [image.shape[1]-150,image.shape[0]]])
   ```

   Resulting in the coordinates:

   |  Source   | Destination |
   | :-------: | :---------: |
   | 140, 720  |  150, 720   |
   | 570, 460  |   150, 0    |
   | 717, 460  |   1130, 0   |
   | 1200, 720 |  1130, 720  |

   Below the lines connecting the vertices are drawn in the raw image and shown in perspective tranformed one:

   

   ![alt text][image4]

   

3. ### Thresholded Binary Image

   At line 29, 7th code cell, `algorithm` is a list of names associated to the available implemented thresholding functions, together with the thresholds parameters. Both names and thresholds can be freely set by the user. 

   At line 60 the function `threshold()` is called, warped image and parameters are passed through and the final binary image `combined` is obtained.

   In this implementation the combination of gradient thresholds and HLS - S channel - has been found successful for the lines identification in each pictures.

   Below an example of a binary warped image.

   

   ![alt text][image3]

   

4. ### Lane-line Identification and Polynomial Fit

   At line 74, 7th code cell, `find_lane_pixels()` is outputing all the information discussed before in Function Definition. Below there is an example of a binary image with fitting lines overlaid:

   

   ![alt text][image5]

   

5. ### Curvature Radius, Lane and Vehicle Position

   - The vehicle position is calculated at line 87, 7th code cell.

     ```python
     offset = lane_pos - image.shape[1]/2
     ```

     It is assumed that the camera is mounted on the center of the car. Thus, the raw image x mid-point `image.shape[1]/2 ` corresponds with the vehicle center position.

   - Offset and fitting coefficients (and y coordinates) are then converted from pixels to meters at line 89, 7th code cell, with `pixel_to_m()`.

     ```
     ploty_cr, left_fit_cr, right_fit_cr, offset_cr = pixel_to_m(ploty, lefty, leftx, righty, rightx, offset)
     ```

   - At line 91, 7th code cell, `left_curverad` and `right_curverad` are then calculated via `measure_curvature_pixels()`.

     ```
     left_curverad, right_curverad = measure_curvature_pixels(ploty_cr, left_fit_cr, right_fit_cr)
     ```

     

6. ### Lane Area Identification

   In picture below, it is shown an example of the pipeline final output. On each image is printed the curvature radius averaged over left and right line curvature and the vehicle position with respect to lane center position:

   

   ![alt text][image6]

   

---

## Pipeline (video processing)

As stated before, the video processing pipeline is slightly different from the test image one. The code is stored in the 9th code cell of the IPython notebook located in "./P2.ipynb".

From line 1 to line 68, 9th code cell, as for the test images pipeline, undistortion, warping and thresholding are performed to the video frame in this case. Afterwards, the steps below are followed:

1. criteria for algorithm used for lines recognition (line 68 to line 69, 9th code cell):

   - `search_around_poly()`:
     - if number of lines detected and stored in lineL_list (or lineR_list) is bigger than the parameter `n_avg`.
     - if previous line (left or right) detected present `.reset` attribute different than 1 (reset not set).

   - `find_lane_pixels()` otherwise.

2. bestx and best_fit averages calculated over goodLine_list left and right (in order to exclude lines that don't pass Sanity Check.

3. Coordinates and offset transformed from pixels to meters via "pixel_to_m()", fit coefficients calculated.

4. Curvature radius + Offset calculation via "measure_curvature_pixels()".

5. Line instance generation based on "detected" information (values stored in "linel" and liner" and subsequently appended in list). If one of the line is not detected, a counter is set, if counter reaches 2, reset flag is set.

6. Sanity Check performed if lines were detected, :

   - Similar curvature
   - Approximately separated by same distance
   - Approximately parallel

   if all three conditions are fulfilled, Sanity Check are passed, counter is reset (.reset attribute = 0), fitting right and left are set to bestx averaged over current good lines detected.

   if any of the conditions is not fulfilled, Sanity Check are not passed, counter keeps counting, fitting right and left are set to bestx averaged over previous good lines detected.

7. Visualization and curvature/ offset information plotted on the raw image.

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

## Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
---
typora-root-url: ./
---

# **Finding Lane Lines on the Road** 

## Writeup

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report
  The output images are in the folder of ![solidWhiteRight](/test_images_output/solidWhiteRight.jpg)
---

### Reflection

### 1. Pipeline Description

My pipeline consisted of 5 steps. 
* Converted the images to grayscale
* Make gaussian blur on the image
* Use canny algorigthm to find the edge
* Mask the region I interested in
* Hightlight the lanes on the image

In order to draw a single line on the left and right lanes, I modified the draw_lines() and the Pipeline function by:
#### Find out all the right lane and left lane from the lines returned by HoughLines()
* If abs value of slop is < 0.5, just ignore it, because it is not big enough to be selected.
* Elif the slop is <= 0, it belongs to left group, because the (0,0) is at the topleft corner. Then put all the points found into array left_line[]
* Elif the slop is > 0, it belongs to right group.Then put all the points found into array right_line[]
* Use np.ployfit() to fit the left lane points and right lane points. It will create a line. Because the two lanes will be vertically between line y = 3/5 hight and y = hight, we use the two y value to get the corresponding x value. These are the start and end points of the two lanes. 
* Then use the draw_lines() to draw a line based on the four start end points.


### 2. Identify potential shortcomings with your current pipeline

* If there is white car near my car, it may recoginze it as a line
* If there is one car in front of my car and blocked the lane, the algorithm might problem identify the lane.


### 3. Suggest possible improvements to your pipeline

* Need to take the shape and size of the lane into consideration
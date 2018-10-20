import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Read in an image
img = cv2.imread('test_image2.png')
nx = 8 # the number of inside corners in x
ny = 6 # the number of inside corners in y

# 临时代码，用来获取undistorted之后，找到的几个corner的坐标
# undist_img = cv2.undistort(img, mtx, dist, None, mtx)
# # 2) Convert to grayscale
# gray = cv2.cvtColor(undist_img, cv2.COLOR_BGR2GRAY)
# # 3) Find the chessboard corners
# ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

# print(corners)

# MODIFY THIS FUNCTION TO GENERATE OUTPUT 
# THAT LOOKS LIKE THE IMAGE ABOVE
def corners_unwarp(img, nx, ny, mtx, dist):
    # Pass in your image into this function
    # Write code to do the following steps
    # 1) Undistort using mtx and dist
    undist_img = cv2.undistort(img, mtx, dist, None, mtx)
    # 2) Convert to grayscale
    gray = cv2.cvtColor(undist_img, cv2.COLOR_BGR2GRAY)
    # 3) Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    # 4) If corners found: 
    if ret:
        # a) draw corners
        cv2.drawChessboardCorners(undist_img, (nx,ny), corners,ret)
        # b) define 4 source points src = np.float32([[,],[,],[,],[,]])
        # 这4个corner是从undistored的图像中找出来的。
        src_points = np.float32([[436.50708, 114.335556],[1103.9984, 224.66327],
                         [465.32727, 768.5894],[1075.5115, 658.0274]])
        #Note: you could pick any four of the detected corners 
        # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
        # 通过查看原图，找到每个block所占有的像素数：168，然后计算出来这4个corner应该的位置
        dst_points = np.float32([[50.0, 50.00],[1226.0, 50.0],
                                [50.0, 890.00],[1226.0, 890.0]])
        # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        # e) use cv2.warpPerspective() to warp your image to a top-down view
        warpped_img = cv2.warpPerspective(undist_img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
        warped = np.copy(warpped_img) 
        return warped, M

top_down, perspective_M = corners_unwarp(img, nx, ny, mtx, dist)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(top_down)
ax2.set_title('Undistorted and Warped Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
print("Done")

#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
from scipy import stats
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    # If there are no lines to draw, exit.
    if lines is None:
        return
    # Make a copy of the original image.
    img = np.copy(img)
    # Create a blank image that matches the original in size.
    line_img = np.zeros(
        (
            img.shape[0],
            img.shape[1],
            3
        ),
        dtype=np.uint8,
    )
    # Loop over all lines and draw them on the blank image.
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    # Merge the image with the lines onto the original.
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    # Return the modified image.
    return img

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images_output directory.
def hightlight_lanes(image):
    #gray_image = grayscale(image)
    blury_grey = gaussian_blur(image, kernel_size=3)
    edges = canny(blury_grey, low_threshold= 50, high_threshold=150)
    height = image.shape[0]
    width = image.shape[1]
    # Next we'll create a masked edges image region_of_interest()
    vertices = np.array([[(0,height),(width/2, height/2), (width,height)]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)

    rho = 6
    theta = np.pi/180
    threshold = 150
    min_line_length = 40
    max_line_gap = 25
    line_image = np.copy(image)*0 

    #lines = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)

    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1) #  Calculating the slope.
            if math.fabs(slope) < 0.5: #  Only consider extreme slope
                continue
            if slope <= 0: #  If the slope is negative, left group.
                left_line_x.extend([x1, x2])
                left_line_y.extend([y1, y2])
            else: #  Otherwise, right group.
                right_line_x.extend([x1, x2])
                right_line_y.extend([y1, y2])
    min_y = image.shape[0] * 3 / 5   # 1/2 is not good enough
    max_y = image.shape[0]  
    if (len(left_line_x) ==0 or len(left_line_y) ==0 or len(right_line_x) == 0 and len(right_line_y) == 0):
        raise Exception("Could not find lane based on current configuration. Please turn the Hough arguments.")
    else:
        poly_left = np.poly1d(np.polyfit(
            left_line_y,
            left_line_x,
            deg=1
        ))
        left_x_start = int(poly_left(max_y))
        left_x_end = int(poly_left(min_y))

        poly_right = np.poly1d(np.polyfit(
            right_line_y,
            right_line_x,
            deg=1
        ))
        right_x_start = int(poly_right(max_y))
        right_x_end = int(poly_right(min_y))

        # slope_left, intercept_left, r_value, p_value, std_err = stats.linregress(left_line_x, left_line_y)
        # slope_right, intercept_right, r_value, p_value, std_err = stats.linregress(right_line_x, right_line_y)
        # left_x_start    =   (min_y - intercept_left)/slope_left
        # left_x_end      =   (max_y - intercept_left)/slope_left
        # right_x_start   =   (min_y - intercept_right)/slope_right
        # right_x_end     =   (min_y - intercept_right)/slope_right

        line_image = draw_lines(
            image,
            [[
                [left_x_start, max_y, left_x_end, min_y],
                [right_x_start, max_y, right_x_end, min_y],
            ]],
            thickness=5,
        )

        color_edges = np.dstack((edges, edges, edges)) 
        combo = weighted_img(line_image, image)
        return combo


def hightlight_lanes2(images):
    #gray_image = grayscale(image)
    blury_grey = gaussian_blur(image, kernel_size=3)
    edges = canny(blury_grey, low_threshold= 50, high_threshold=150)
    height = image.shape[0]
    width = image.shape[1]
    # Next we'll create a masked edges image region_of_interest()
    vertices = np.array([[(0,height),(width/2, height/2), (width,height)]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)

    rho = 6
    theta = np.pi/180
    threshold = 150
    min_line_length = 40
    max_line_gap = 25
    line_image = np.copy(image)*0 

    #lines = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)

    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1) #  Calculating the slope.
            if math.fabs(slope) < 0.5: #  Only consider extreme slope
                continue
            if slope <= 0: #  If the slope is negative, left group.
                left_line_x.extend([x1, x2])
                left_line_y.extend([y1, y2])
            else: #  Otherwise, right group.
                right_line_x.extend([x1, x2])
                right_line_y.extend([y1, y2])
    min_y = image.shape[0] * 3 / 5   # 1/2 is not good enough
    max_y = image.shape[0]  
    if (len(left_line_x) ==0 or len(left_line_y) ==0 or len(right_line_x) == 0 and len(right_line_y) == 0):
        raise Exception("Could not find lane based on current configuration. Please turn the Hough arguments.")
    else:
        poly_left = np.poly1d(np.polyfit(
            left_line_y,
            left_line_x,
            deg=1
        ))
        left_x_start = int(poly_left(max_y))
        left_x_end = int(poly_left(min_y))

        poly_right = np.poly1d(np.polyfit(
            right_line_y,
            right_line_x,
            deg=1
        ))
        right_x_start = int(poly_right(max_y))
        right_x_end = int(poly_right(min_y))

        # slope_left, intercept_left, r_value, p_value, std_err = stats.linregress(left_line_x, left_line_y)
        # slope_right, intercept_right, r_value, p_value, std_err = stats.linregress(right_line_x, right_line_y)
        # left_x_start    =   (min_y - intercept_left)/slope_left
        # left_x_end      =   (max_y - intercept_left)/slope_left
        # right_x_start   =   (min_y - intercept_right)/slope_right
        # right_x_end     =   (min_y - intercept_right)/slope_right

        line_image = draw_lines(
            image,
            [[
                [left_x_start, max_y, left_x_end, min_y],
                [right_x_start, max_y, right_x_end, min_y],
            ]],
            thickness=5,
        )

        color_edges = np.dstack((edges, edges, edges)) 
        combo = weighted_img(line_image, image)
        return combo

def save_lane_hightlighted_images(input_image_path):
    image = mpimg.imread(input_image_path)
    plt.imshow(hightlight_lanes(image))
    out_image = os.path.join("./test_images_output/", str(os.path.basename(input_image_path)))
    plt.savefig(out_image)


def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    result = hightlight_lanes(image)
    return result

def process_test_images():
    for each in os.listdir('./test_images') :
        save_lane_hightlighted_images(os.path.join('./test_images', each))

white_output = 'test_videos_output/solidWhiteRight.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

#process_test_images()

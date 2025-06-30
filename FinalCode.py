import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
from CannyEdgeDetection import canny_edge
from HoughTransform import hough_lines_detection


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)

    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[128, 0, 128], thickness=10):
    """
    This function draws lines with the specified color and thickness.
    The color is set to purple (RGB: 128, 0, 128) for the lane lines.
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def slope_lines(image, lines):
    img = image.copy()
    poly_vertices = []
    order = [0, 1, 3, 2]

    left_lines = []  # Like /
    right_lines = []  # Like \
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                pass  # Vertical Lines
            else:
                m = (y2 - y1) / (x2 - x1)
                c = y1 - m * x1

                if m < 0:
                    left_lines.append((m, c))
                elif m >= 0:
                    right_lines.append((m, c))

    left_line = np.mean(left_lines, axis=0)
    right_line = np.mean(right_lines, axis=0)

    for slope, intercept in [left_line, right_line]:
        rows, cols = image.shape[:2]
        y1 = int(rows)
        y2 = int(rows * 0.6)
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        poly_vertices.append((x1, y1))
        poly_vertices.append((x2, y2))
        draw_lines(img, np.array([[[x1, y1, x2, y2]]]), color=[128, 0, 128])  # Purple lines

    poly_vertices = [poly_vertices[i] for i in order]
    cv2.fillPoly(img, pts=np.array([poly_vertices], 'int32'), color=(255, 255, 0))  # Yellow pipeline
    return cv2.addWeighted(image, 0.7, img, 0.4, 0.)


def get_vertices(image):  #NOVA
    rows, cols = image.shape[:2]
    bottom_left = [cols * 0.15, rows]
    top_left = [cols * 0.45, rows * 0.6]
    bottom_right = [cols * 0.95, rows]
    top_right = [cols * 0.55, rows * 0.6]

    ver = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return ver


def lane_finding_pipeline(img):
    canny_image = canny_edge(img)
    masked_img = region_of_interest(img=canny_image, vertices=get_vertices(img))
    hough_lines = hough_lines_detection(img=masked_img, rho=1, theta=np.pi / 180, threshold=20, min_line_len=20,
                                max_line_gap=180)
    line_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    line_img = slope_lines(line_img, hough_lines)
    final = cv2.addWeighted(img, 0.8, line_img, 1., 0.)

    return final


for image_path in list(os.listdir('./test_images')):
    image = mpimg.imread(f'./test_images/{image_path}')
    processed_image = lane_finding_pipeline(image)

    plt.figure(figsize=(10, 5))
    plt.imshow(processed_image)
    plt.xticks([]), plt.yticks([])
    plt.title("Output Image [Lane Line Detected]")
    plt.show()

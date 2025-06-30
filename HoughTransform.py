import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from CannyEdgeDetection import canny_edge
from RegionOfInterest import region_of_interest


def hough_lines_detection(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                            min_line_len, max_line_gap)
    return lines


def draw_lines(img, lines, color=[0, 255, 0], thickness=10):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


if __name__ == "__main__":
    image = mpimg.imread('test_images/solidYellowLeft.jpg')

    original_image = np.copy(image)

    # ROI
    imshape = image.shape
    vertices = np.array([[(0, imshape[0]), (450, 290), (490, 290), (imshape[1], imshape[0])]], dtype=np.int32)
    edges = canny_edge(image)
    masked_edges = region_of_interest(edges, vertices)

    rho = 1
    theta = np.pi / 180
    threshold = 2
    min_line_length = 4
    max_line_gap = 5

    lines = hough_lines_detection(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)

    line_image = np.zeros_like(original_image)  # blank image
    draw_lines(line_image, lines, color=[0, 255, 0], thickness=10)

    plt.imshow(image)
    plt.title("Input Image")
    plt.show()

    # Overlay the lines on the original image
    lines_on_original = cv2.addWeighted(original_image, 0.8, line_image, 1, 0)
    plt.imshow(lines_on_original)
    plt.xticks([]), plt.yticks([])
    plt.title("Green Lane Line")
    plt.show()

    # Draw the region of interest in purple on the original image
    cv2.polylines(lines_on_original, vertices, True, (255, 0, 255), 10)  # Purple for region of interest
    plt.imshow(lines_on_original)
    plt.title("Green Lane Line and Purple Region of Interest on Original Image")
    plt.show()

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2


def canny_edge(image):
    # конверзија во grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # дефинирање на кернел големина за Gaussian blurring
    kernel_size = 5  # мора да е прост број (3, 5, 7...)
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    low_threshold = 180
    high_threshold = 240
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    return edges


if __name__ == "__main__":
    image = mpimg.imread('test_images/solidYellowLeft.jpg')
    color_select = np.copy(image)
    canny_image = canny_edge(color_select)
    plt.imshow(canny_image, cmap='Greys_r')
    plt.title("Canny Edge Detection Image")
    plt.show()

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def defining_color_criteria(image):
    red_threshold = 200
    green_threshold = 200
    blue_threshold = 200

    rgb_threshold = [red_threshold, green_threshold, blue_threshold]
    # идентификување на пиксели под thresholds
    thresholds = (image[:, :, 0] < rgb_threshold[0]) \
                 | (image[:, :, 1] < rgb_threshold[1]) \
                 | (image[:, :, 2] < rgb_threshold[2])
    color_select[thresholds] = [0, 0, 0]
    return image


if __name__ == "__main__":
    image = mpimg.imread('test_images/solidWhiteCurve.jpg')
    color_select = np.copy(image)
    defining_color_criteria(color_select)

    plt.imshow(image)
    plt.title("Input Image")
    plt.show()
    plt.imshow(color_select)
    plt.title("Color Selected Image")
    plt.show()

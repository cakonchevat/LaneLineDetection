import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)  # Same size as the image but filled with zeros (black)

    # Fill the polygon in the mask with white
    cv2.fillPoly(mask, vertices, (255, 255, 255))

    # Apply the mask to the input image: only show the region inside the polygon
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


if __name__ == "__main__":
    image = mpimg.imread('test_images/solidYellowLeft.jpg')

    imshape = image.shape
    vertices = np.array([[(0, imshape[0]), (450, 290), (490, 290), (imshape[1], imshape[0])]], dtype=np.int32)

    masked_image = region_of_interest(image, vertices)
    plt.imshow(masked_image)
    plt.title("Masked Image with ROI")
    plt.show()

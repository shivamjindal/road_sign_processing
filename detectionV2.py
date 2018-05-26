"""
FILE: detection.py
AUTHOR: Shivam Jindal

For New York University's Autonomous Vehicle Team -- IGVC 2018
- Program to recognize what a given sign is
"""

import cv2
import numpy as np
from PIL import Image
from pytesseract import image_to_string
import matplotlib.pyplot as plt


def get_dominant_color(img):
    """"""
    Z = img.reshape((-1, 3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 1
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    res2 = cv2.resize(res2, (300,300))
    return res2[0][0]

def is_dom_red(color):
    """
    ***NOTE: This needs to be updated and improved upon later

    Currently, this just checks to see if the red value in an image is the most prominent
    :param color: bgr values of the provided image
    :return: True if the dominant color is red. False otherwise
    """
    if color[2] > color[0] and color[2] > color[1]:
        return True
    else:
        return False

def read_sign(image):
    """
    Uses tesseract library to determine the words on an image --
    - this works best with no turns signs and road closed signs
    
    :param image: the filepath for an image
    :return: the text in a given image
    """

    return(image_to_string(Image.open(image),lang='eng'))


def get_direction(image):
    """
    Split image up in half
    If there are more activated pixels on left half, it points left.
    Else, it points right.
    :param image: path to a given image (should only be for one way signs)
    :return: Direction of the one way sign
    """
    img = cv2.imread(image, 0)
    ret, img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

    h, w = img.shape

    left_half = img[0:h, 0:int(w/2)]
    right_half = img[0:h, int(w/2):w]

    if cv2.countNonZero(left_half) > cv2.countNonZero(right_half):
        return "LEFT"
    else:
        return "RIGHT"


def get_match(image):
    img = cv2.imread(image)
    dom_color = get_dominant_color(img)

    if is_dom_red(dom_color):
        return "STOP"
    else:
        sign_words = read_sign(image).replace("\n", " ").strip()
        sign_words = sign_words.replace("0", "O")
        if (sign_words == "NO TURNS" or sign_words == "ROAD CLOSED"):
            return sign_words.replace("\n", " ")
        else:
            return get_direction(image)

if __name__ == "__main__":
    test_image = "Images/road_closed.png" #place filepath of image here

    print("TESTING FOR: \t", test_image, '\n=========')

    print(get_match(test_image))

import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
Given an image detected to be a sign, this will recognize what kind of sign it is
Note to whoever uses this afterwards: 
    - make each image the same size of your desire and make each query image that same size
        - this should get you better results    

Potential Improvements: 
    -  preprocess the query images to get rid of potential noise that may occur. 
    -  guarantee better accuracy by also checking the primary color?

"""


def get_match(image, train_list):
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    image = cv2.imread(image)
    image = cv2.morphologyEx(
        image,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    )
    image = cv2.morphologyEx(
        image,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    )

    matches = []

    orb = cv2.ORB_create() # orb is a way to get the key points and descriptors of an image
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)   # bfmatcher is a way to match key points and descriptors

    # just a counter
    index = 0

    for img in train_list:

        #get the key points and descriptors for the actual sign and tested image
        kp, des = orb.detectAndCompute(img, None)                   # train
        kp_test, des_test = orb.detectAndCompute(image, None)       # query

        # match - list of all the matches
        matches.append(sorted(bf.match(des, des_test), key=lambda x: x.distance))


        # draws the train and query image and the matches
        img2 = cv2.drawMatchesKnn(img, kp, image, kp_test, matches[:10], None, flags=2)

        # lets you see each image and its matches
        # cv2.imshow('show', img2)
        # cv2.waitKey(0)  # note you have to press a button in order to move to the next image
        index+=1


    # this block basically gets you the closest match
    # (the lower the score the more likely the match)
    unique_signs = ["stop", "no turns", "one way right", "one way left", "road closed"]
    index = 0
    # so this takes the average of first 10 in each match list and prints them out
    for i in matches:
        distance_list = []
        for j in i[:10]:
            distance_list.append(j.distance)
        try:
            print(unique_signs[index], sum(distance_list)/len(distance_list))
        except ZeroDivisionError:
            print("-")
        index += 1



if __name__ == "__main__":
    image_to_match = "Images/one_way_right_test_1.png" #place filepath of image here
    # image_to_match = "Images/one_way_left_test4.jpg" #place filepath of image here
    # image_to_match = "Images/stop.jpg" #place filepath of image here

    cv2.namedWindow("stop", cv2.WINDOW_NORMAL)
    stop = cv2.imread("Images/stop.jpg")
    stop = cv2.resize(stop, (300,300))

    cv2.namedWindow("no_turns", cv2.WINDOW_NORMAL)
    no_turns = cv2.imread("Images/no_turns.jpg")
    no_turns = cv2.resize(no_turns,(300,300))


    cv2.namedWindow("one_way_left", cv2.WINDOW_NORMAL)
    one_way_left = cv2.imread("Images/one_way_left.png")
    # one_way_left = cv2.resize(one_way_left,None, fx = .6, fy=.6)

    cv2.namedWindow("one_way_right", cv2.WINDOW_NORMAL)
    one_way_right = cv2.imread("Images/one_way_right.jpg")
    one_way_right = cv2.resize(one_way_right, None, fx = .8, fy=.8)

    cv2.namedWindow("road_closed", cv2.WINDOW_NORMAL)
    road_closed = cv2.imread("Images/road_closed.png")
    road_closed = cv2.resize(road_closed, None, fx = .5, fy=.5)

    train_image_list = [stop, no_turns, one_way_right, one_way_left, road_closed]

    print("TESTING FOR: \t", image_to_match, '\n========= \nNOTE: THE LOWER THE SCORE, THE MORE LIKELY IT IS THAT SIGN\n')
    get_match(image_to_match, train_image_list)

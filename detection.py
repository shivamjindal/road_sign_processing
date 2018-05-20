import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_match(image, train_list):
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    image = cv2.imread(image)
    image = cv2.resize(image, (500, 300))

    match_lists = []

    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # just a counter
    index = 0

    for img in train_list:

        #get the key points and descriptors for the actual sign and tested image
        kp, des = orb.detectAndCompute(img, None)
        kp_test, des_test = orb.detectAndCompute(image, None)

        # match_lists - list of how each image did compared to each sign
        match_lists.append(sorted(bf.match(des, des_test), key = lambda x:x.distance))

        # image with the matches drawn
        img2 = cv2.drawMatches(img, kp, image, kp_test, match_lists[index][:25], None, flags=2)

        # lets you see each image and it's matches
        cv2.imshow('show', img2)
        cv2.waitKey(0)

        index+=1

    list = ["stop", "no turns", "one way left", "one way right", "road closed"]
    index = 0
    # so this takes the average of first 10 in each match list and prints them out
    for i in match_lists:
        the_list = []
        for j in i[:25]:
            the_list.append(j.distance)
        print(list[index], sum(the_list)/len(the_list))
        index += 1
        # print(i[0].distance)





if __name__ == "__main__":
    image_to_match = "one_way_left_test4.jpg" #place filepath of image here

    cv2.namedWindow("stop", cv2.WINDOW_NORMAL)
    stop = cv2.imread("stop.jpg")
    stop = cv2.resize(stop, (250, 250))

    # stop = cv2.Canny(stop, 100,200)

    cv2.namedWindow("no_turns", cv2.WINDOW_NORMAL)
    no_turns = cv2.imread("no_turns.jpg")
    no_turns = cv2.resize(no_turns, (100, 100))
    # no_turns = cv2.Canny(no_turns, 200,200)


    cv2.namedWindow("one_way_left", cv2.WINDOW_NORMAL)
    one_way_left = cv2.imread("one_way_left.png")
    one_way_left = cv2.resize(one_way_left, (300, 300))
    # one_way_left = cv2.Canny(one_way_left, 200,200)


    cv2.namedWindow("one_way_right", cv2.WINDOW_NORMAL)
    one_way_right = cv2.imread("one_way_right.jpg")
    one_way_right = cv2.resize(one_way_right, (300, 300))
    # one_way_right = cv2.Canny(one_way_right, 200,200)


    cv2.namedWindow("road_closed", cv2.WINDOW_NORMAL)
    road_closed = cv2.imread("road_closed.png")
    road_closed = cv2.resize(road_closed, (100, 100))
    # road_closed = cv2.Canny(road_closed, 100,100)


    train_image_list = [stop, no_turns, one_way_right, one_way_left, road_closed]
    print(image_to_match, '\n=========')
    get_match(image_to_match, train_image_list)

    # cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    # image_to_match = cv2.imread(image_to_match)
    # image_to_match = cv2.resize(image_to_match, (600, 600))
    # image = cv2.Canny(image_to_match, 1000,200)
    # cv2.imshow("image",image)
    # cv2.waitKey(0)
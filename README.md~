# road_sign_processing

For New York University's Autnomous Vehicle Team - IGVC 2018

Classifies a detected sign as either a stop, one way left, one way right, no turns, or road closed sign.

## Contributors ##
Shivam Jindal

## How it Works (V2) ##

It first tries to simply see the main color within the image. If it is a red-like color, it is classified as a stop sign. If not, it then tries to read the sign. If it reads "NO TURNS" or "ROAD CLOSED", then that sign is classified. For one way signs, it splits the image in half and detects the number of activated pixels in both half. If the left half has more activated pixels, then the one way sign is pointing to the left. Otherwise, it points to the right.

## How it Works (V1) ##

The program finds the keypoints and descriptors of each type of sign using OpenCV's ORB function. From there, OpenCV's Brute-Force Matcher finds the most similar and closest match from the different possible signs to the query sign. 

This process is repeated with each possible sign and the average hamming distances are then calculated. Whichever sign that has the lowest average hamming distance for its first 10 matches is the recognized sign. 


## Installation ##
* pytesseract
* opencv (3.4.1)
* numpy (1.13.3)

## Improvements ##
...To be completed


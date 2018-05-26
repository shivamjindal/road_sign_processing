# road_sign_processing

For New York University's Autnomous Vehicle Team - IGVC 2018

Classifies a detected sign as either a stop, one way left, one way right, no turns, or road closed sign.

## How it Works (V2) ##

It first tries to simply see the main color within the image. If it is a red-like color, it is classified as a stop sign. If not, it then tries to read the sign. If it reads "NO TURNS" or "ROAD CLOSED", then that sign is classified. For one way signs, it splits the image in half and detects the number of activated pixels in both half. If the left half has more activated pixels, then the one way sign is pointing to the left. Otherwise, it points to the right.

## How it Works (V1) ##

~To be completed~

## Installation ##
* pytesseract
* opencv (3.4.1)
* numpy (1.13.3)

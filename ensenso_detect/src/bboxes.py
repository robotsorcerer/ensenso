"""

Author: Olalekan Ogunmolu

This code loads all images in the specified path given
and attempts to ease the process of image labeling
for computer vision classifier builds

Date: May 08, 2017

"""


# import the necessary packages
from __future__ import print_function
import os, cv2, argparse
from os import listdir
from PIL import Image
import numpy as np
import sys
import json

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False, help="Path to the image")
ap.add_argument("-v", "--verbose", type=bool, default=True)
args = vars(ap.parse_args())

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False
mode = None # 'l' for left eye, 'r' for right_eye, 'm' for face
rect = None # opencv rectangle that contains the face or eyes roi
detect = {}

detect_items = ['faces', 'left_eyes', 'right_eyes', 'fore_head', 'nose']

detect_mode = detect_items[2]
detect[detect_mode] = []
#
# for i in range (len(detect_items)):
#     detect[detect_items[i]] = []

file_name, file_image = None, None

def exchange(A, idx1, idx2):
    A[idx1], A[idx2] = A[idx2], A[idx1]

def quicksort(A, p, r):
    if(p < r):
        q = partition(A,p,r)
        quicksort(A, p, q-1)
        quicksort(A, q+1, r)

def partition(A, p, r):
    x = A[r]
    i = p -1
    for j in xrange(p,r):
        if (A[j] <= x):
            i = i + 1
            exchange(A, i, j)
    exchange(A, i+1, r)
    return i + 1

def loadAllImages(images_path):
    image_file_names = []
    images_list = []
    imagesList = listdir(images_path)

    for file_name in imagesList:
        image_file_names.append(file_name)
        image = cv2.imread(images_path + '/' + file_name, 1)
        images_list.append(image)

    #sort the filenames numerically
    p, r = 0, len(image_file_names)
    quicksort(image_file_names,0, r-1)

    return image_file_names, images_list

def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping
    global file_name, file_image
    global detect, mode, rect

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        cropping = False

        # draw a rectangle around the region of interest
        rect = cv2.rectangle(file_image, refPt[0], refPt[1], (0, 255, 0), 2)

def main():
    cwd = os.getcwd()
    images_path = cwd + "/" + "../" + "manikin/raw/face_images"

    file_names, images_list = loadAllImages(images_path)
    global file_name, file_image, rect, detect_mode

    for i in range(len(images_list)):
        file_name, file_image = file_names[i], images_list[i]

        winName = images_path + '/' + file_names[i]

        #we always start with faces, then do all the eyes
        #
        # mode = raw_input("please enter mode of clicks <f, l, or r> ...")
        # if mode == "f":
        #     detect_mode = "faces"
        # elif mode =="l":
        #     detect_mode = "left_eye"
        # elif mode == "r":
        #     detect_mode = "right_eye"

        clone = images_list[i].copy()
        cv2.namedWindow(file_name)
        cv2.setMouseCallback(file_name, click_and_crop)

        def save(file_name):
            x1, y1, x4, y4 = refPt[0][0], refPt[0][1], refPt[1][0], refPt[1][1]
            x2, y2, x3, y3 = x4, y1, x1, y4
            top_left, top_right = [x1, y1], [x2, y2]
            bot_left, bot_right = [x3, y3], [x4, y4]
            #
            detect[detect_mode].append({
                file_name: (top_left + top_right, bot_left + bot_right)
                })

            # dump detect bounding boxes to a json file
            with open('data/' + detect_mode + '_bboxes.txt', 'w') as outfile:
                json.dump(detect, outfile)

            #show what you got
            print(detect)

        while True:
            # display the image and wait for a keypress
            cv2.imshow(file_name, images_list[i])
            key = cv2.waitKey(1) & 0xFF

            # if the 'r' key is pressed, reset the cropping region
            if key == ord("r"):
                images_list[i] = clone.copy()
                file_name, file_image = file_names[i], images_list[i]

            # if the 'n' key is pressed, break from the loop
            elif key == ord("n"):
                save(file_name)
                cv2.destroyWindow(file_name)
                break

            if key == ord("s"):
                save()

    # close all open windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

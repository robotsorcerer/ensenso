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
detect_items = ['faces', 'left_eyes', 'right_eyes']
detect_mode = []

for i in range (len(detect_items)):
    detect[detect_items[i]] = []

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
    global file_name, file_image, rect

    for i in range(len(images_list)):
        file_name, file_image = file_names[i], images_list[i]

        file_names[i] = images_path + '/' + file_names[i]

        mode = raw_input("please enter mode of clicks <f, l, or r> ...")
        if mode == "f":
            detect_mode = "faces"
        elif mode =="l":
            detect_mode = "left_eye"
        elif mode == "r":
            detect_mode = "right_eye"

        clone = images_list[i].copy()
        cv2.namedWindow(file_names[i])
        cv2.setMouseCallback(file_names[i], click_and_crop)

        while True:
            # display the image and wait for a keypress
            # cv2.resize(images_list[i], images_list[i], 0.5, 0.5)
            cv2.imshow(file_names[i], images_list[i])
            key = cv2.waitKey(1) & 0xFF

            # if the 'r' key is pressed, reset the cropping region
            if key == ord("r"):
                images_list[i] = clone.copy()
                file_name, file_image = file_names[i], images_list[i]

            # if the 'n' key is pressed, break from the loop
            elif key == ord("n"):
                cv2.destroyWindow(file_names[i])
                break

        x1, y1, x4, y4 = refPt[0][0], refPt[0][1], refPt[1][0], refPt[1][1],
        x2, y2, x3, y3 = x4, y1, x1, y4
        top_left, top_right = [x1, y1], [x2, y2]
        bot_left, bot_right = [x3, y3], [x4, y4]
        #
        detect[detect_mode].append({
            file_name: (top_left + top_right, bot_left + bot_right)
            })

        print(detect)

        # print(x, y, w, h)

    # if there are two reference points, then crop the region of interest
    # from the image and display it
    if len(refPt) == 2:
        roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
        cv2.imshow("ROI", roi)
        cv2.waitKey(0)

    # close all open windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

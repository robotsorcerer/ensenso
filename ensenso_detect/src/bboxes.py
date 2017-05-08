# import the necessary packages
from __future__ import print_function
import os, cv2, argparse
from os import listdir
from PIL import Image
import numpy as np
import sys

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False, help="Path to the image")
ap.add_argument("-v", "--verbose", type=bool, default=True)
args = vars(ap.parse_args())

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False

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
        # image = Image.open(images_path + '/' + file_name)
        # cv_image = cv2.cvtColor(np.array(image),  cv2.COLOR_RGB2BGR)
        images_list.append(image)

    #sort the filenames numerically
    p, r = 0, len(image_file_names)
    quicksort(image_file_names,0, r-1)

    return image_file_names, images_list

def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping

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

        print(file_name)
        # draw a rectangle around the region of interest
        cv2.rectangle(file_image, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow(file_name, file_image)

file_name, file_image = None, None

def main(i):
    images_path = "/home/lex/catkin_ws/src/sensors/ensenso_detect/manikin/raw/face_images"

    file_names, images_list = loadAllImages(images_path)
    if args['verbose']:
        # print(len(images_list))
        # for i in file_names:
            # print(i)
        file_names[1] = images_path + '/' + file_names[1]
        # print('file_names[1]', file_names[1])
        # print(images_list[1], 'images_list[1]')
        # cv2.namedWindow(file_names[1])
        # cv2.imshow(file_names[1], images_list[1])

    key = cv2.waitKey(1) & 0xFF

    # for i in range(len(images_list)):
    # while True:
    file_name, file_image = file_names[i], images_list[i]

    file_names[i] = images_path + '/' + file_names[i]

    clone = images_list[i].copy()
    cv2.namedWindow(file_names[i])
    cv2.setMouseCallback(file_names[i], click_and_crop)

    # display the image and wait for a keypress
    cv2.imshow(file_names[i], images_list[i])

    # if the 'r' key is pressed, reset the cropping region
    if key == ord("r"):
        images_list[i] = clone.copy()

    if(key == ord("n")):
        cv2.destroyWindow(file_names[i])
        i = i +1



    # if there are two reference points, then crop the region of interest
    # from the image and display it
    if len(refPt) == 2:
        roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
        cv2.imshow("ROI", roi)
        cv2.waitKey(0)

    # close all open windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    i = 0
    while True:
        main(i)
        raw_input("press any key ...")
        i = i +1

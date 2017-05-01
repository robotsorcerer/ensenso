#!/usr/bin/env python
# image_processing.py is bare-bones subscriber, in Object Oriented form.
# If something is publishing to /camera/image_mono, it receives that
# published image and writes "image received".
# To run, use roslaunch on camera.launch or <bagfile>.launch and then,
# in another terminal, type "python image_processing.py"
# Used in Lab 5 of BE 107 at Caltech
# By Melissa Tanner, mmtanner@caltech.edu, April 2015

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class Image_Processor:
    def __init__(self):
       self.image_source = "/ensenso/image_combo"
       self.cvbridge = CvBridge()
       self.counter = 0

       # Raw Image Subscriber
       self.image_sub = rospy.Subscriber(self.image_source,Image,self.image_callback)

    def image_callback(self, rosimg):
        print("image recieved")
        self.counter +=1
        if self.counter%10 is 0:
            # Convert the image.
            try:
                 # might need to change to bgr for color cameras
                img = self.cvbridge.imgmsg_to_cv2(rosimg, 'passthrough')
            except CvBridgeError:
                rospy.logwarn ('Exception converting background image from ROS to opencv:  ')
                img = np.zeros((320,240))

            #Display
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.imshow('image', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()



################################################################################
def main():
  image_processor = Image_Processor()
  try:
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.DestroyAllWindows()

################################################################################
if __name__ == '__main__':
    rospy.init_node('image_processor', anonymous=True)
    main()

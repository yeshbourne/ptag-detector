# import image utilities
import os
import numpy as np
import cv2

images_path = './images/temp'


# define a function that rotates images in the current directory
# given the rotation in degrees as a parameter
def rotateImages():
  for f_name in os.listdir(images_path):
    file_path = os.path.normpath(os.path.join(images_path, f_name))
    img = cv2.imread(file_path)
    img_rotate_90_clockwise = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(file_path, img_rotate_90_clockwise)

rotateImages()

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import cv2
import copy
import sys

PLUSINF = sys.float_info.max
MINUSINF = sys.float_info.min
np.set_printoptions(precision=3)

projdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    ipath1 = projdir.replace('\\', '/') + '/data/SIFT1_img.jpg'
    ipath2 = projdir.replace('\\', '/') + '/data/SIFT2_img.jpg'

    src_image1 = cv2.imread(ipath1)
    src_image2 = cv2.imread(ipath2)

    image1 = cv2.cvtColor(src_image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(src_image2, cv2.COLOR_BGR2RGB)

    sift = cv2.SIFT_create()

    kp1 = sift.detect(image1,None)
    kpImage1 = cv2.drawKeypoints(image1,kp1,src_image1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(kpImage1),plt.show()

    kp2 = sift.detect(image2,None)
    kpImage2 = cv2.drawKeypoints(image2,kp2,src_image2,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(kpImage2),plt.show()

    keypoints_1, descriptors_1 = sift.detectAndCompute(image1,None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(image2,None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    matches = bf.match(descriptors_1,descriptors_2)
    matches = sorted(matches, key = lambda x:x.distance)

    kpMatchedImage = cv2.drawMatches(image1, keypoints_1, image2, keypoints_2, matches[:50], None, matchColor=(255, 0, 0),flags=2)
    plt.imshow(kpMatchedImage),plt.show()


from __future__ import print_function
import cv2 
import argparse
import numpy as np
# VideoCapture オブジェクトを取得します
from processing import detectAndDisplay

def main(args):
    camera_device = args.camera
    #-- 2. Read the video stream
    cap = cv2.VideoCapture(camera_device)
    if not cap.isOpened:
        print('--(!)Error opening video capture')
        exit(0)
    for i in range(1000):
        mask_img = cv2.imread(args.mask_img,)
        ret, frame = cap.read()
        if frame is None:
            print('--(!) No captured frame -- Break!')
            break
        detectAndDisplay(frame, mask_img, args, i)
        if cv2.waitKey(10) == 27:
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
    parser.add_argument('--face_cascade', help='Path to face cascade.', default='data/haarcascades/haarcascade_frontalface_alt.xml')
    parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
    parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
    parser.add_argument('--mask_img', help='path to mask image', default='/home/ryota/face_detection/src/data/mask_image/drink_tapioka_tea_schoolboy.png')
    args = parser.parse_args()

    main(args)
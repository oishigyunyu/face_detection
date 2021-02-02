from __future__ import print_function
import cv2 
import argparse
import numpy as np
# VideoCapture オブジェクトを取得します

def detectAndDisplay(frame, mask_img):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    print(mask_img.shape)
    mask_h = mask_img.shape[0]
    mask_w = mask_img.shape[1]
    
    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        #mat = cv2.getRotationMatrix2D((w / 2, h / 2), 45, 0.5)
        mat = np.array([[1.0, 0.0, center[0] - mask_h//4],
                        [0.0, 1.0, center[1] - mask_w//4]]).reshape(2,3)
        #frame = cv2.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
        mask_img = cv2.resize(mask_img, (mask_h//2, mask_w//2))
        frame = cv2.warpAffine(mask_img, M=mat, dsize=(frame.shape[1], frame.shape[0]), borderMode=cv2.BORDER_TRANSPARENT, dst=frame,)
        faceROI = frame_gray[y:y+h,x:x+w]
        #-- In each face, detect eyes
    cv2.imshow('Capture - Face detection', frame)

parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--face_cascade', help='Path to face cascade.', default='data/haarcascades/haarcascade_frontalface_alt.xml')
parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
parser.add_argument('--mask_img', help='path to mask image', default='/home/ryota/face_detection/src/data/mask_image/drink_tapioka_tea_schoolboy.png')
args = parser.parse_args()
face_cascade_name = args.face_cascade
eyes_cascade_name = args.eyes_cascade
face_cascade = cv2.CascadeClassifier()
eyes_cascade = cv2.CascadeClassifier()
#-- 1. Load the cascades
if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv2.samples.findFile(eyes_cascade_name)):
    print('--(!)Error loading eyes cascade')
    exit(0)
camera_device = args.camera
#-- 2. Read the video stream
cap = cv2.VideoCapture(camera_device)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)
while True:

    print(args.mask_img)
    mask_img = cv2.imread(args.mask_img,)
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    detectAndDisplay(frame, mask_img)
    if cv2.waitKey(10) == 27:
        break
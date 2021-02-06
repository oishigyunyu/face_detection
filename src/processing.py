import cv2 
import numpy as np

def detectAndDisplay(frame, mask_img, args, i):
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
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    mask_h = mask_img.shape[0]
    mask_w = mask_img.shape[1]
    
    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        #translation matrix to put mask into the center of the face
        mat = np.array([[1.0, 0.0, center[0] - mask_h//4],
                        [0.0, 1.0, center[1] - mask_w//4]]).reshape(2,3)
        #frame = cv2.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
        mask_img = cv2.resize(mask_img, (mask_h//2, mask_w//2))
        frame = cv2.warpAffine(mask_img, M=mat, dsize=(frame.shape[1], frame.shape[0]), borderMode=cv2.BORDER_TRANSPARENT, dst=frame,)
        faceROI = frame_gray[y:y+h,x:x+w]
        #-- In each face, detect eyes
    cv2.imshow('Capture - Face detection', frame)
    cv2.imwrite(str(i) + '.png', frame)
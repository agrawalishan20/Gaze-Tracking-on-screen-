import numpy as np
import cv2
import pyautogui
import os
import time
import dlib

def getEye(cap, times = 1, coords = (0,0), counterStart = 0, folder = "eyes1"):
    os.makedirs(folder, exist_ok=True)
    counter = counterStart
    ims = []
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    while counter < counterStart+times:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        for face in faces:
            landmarks = predictor(gray, face)
            # Gaze detection
            left_eye_region = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                        (landmarks.part(37).x, landmarks.part(37).y),
                                        (landmarks.part(38).x, landmarks.part(38).y),
                                        (landmarks.part(39).x, landmarks.part(39).y),
                                        (landmarks.part(40).x, landmarks.part(40).y),
                                        (landmarks.part(41).x, landmarks.part(41).y)], np.int32)
            min_x = np.min(left_eye_region[:, 0])
            max_x = np.max(left_eye_region[:, 0])
            min_y = np.min(left_eye_region[:, 1])
            max_y = np.max(left_eye_region[:, 1])
            var = 8
            eye = gray[min_y - var: max_y + var, min_x - var: max_x + var]
            left_eye = cv2.resize(eye, dsize=(100, 50))
            #cv2.imshow("Eye", eye1)
            cv2.imwrite(
                    folder + "/" + str(coords[0]) + "." + str(coords[1]) + "." + str(
                        counter) + ".jpg", left_eye)
        counter += 1




cap = cv2.VideoCapture(0)
#for i in [100, 1600]:
 #   for j in [200, 800]:
for i in [20, 480, 1200, 1900]:
   for j in [10, 270, 650, 1000]:
        print(i,j)
        pyautogui.moveTo(i, j)
        getEye(cap, times = 100, coords=(i,j), counterStart=0, folder = "eyes_train")
cap.release()
cv2.destroyAllWindows()

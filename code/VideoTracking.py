import os
import cv2 as cv
import numpy as np
import copy
import torch
import torch.nn as nn
import pyautogui
import matplotlib.pyplot as plt
import time
import dlib


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#print(torch.cuda.is_available())

class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        f2 = 4
        self.layer2 = nn.Sequential(
            nn.Conv2d(1, f2, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(f2),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(50 * 25 * f2, 200)
        self.fc2 = nn.Linear(200, 20)
        self.fc3 = nn.Linear(20, 1)



    def forward(self,x):
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


def eyetrack(xshift = 30, yshift=150, frameShrink = 0.15):
    # X classifiers
    sixn = ConvNet().to(device)
    sixn.load_state_dict(torch.load("xModels/102.plt",map_location=device))
    sixn.eval()

    """sevent = venty().to(device)
    sevent.load_state_dict(torch.load("xModels/70test.plt",map_location=device))
    sevent.eval()"""

    #If u want to create ensembles of either x and y hit the code above and add the object name in modList
    def ensembleX(im):  # 58 accuracy
        modList = [sixn]
        sumn = 0
        for mod in modList:
            sumn += mod(im).item()
        return sumn / len(modList)

    # Y classifiers
    fiv = ConvNet().to(device)
    fiv.load_state_dict(torch.load("yModels/56.plt",map_location=device))
    fiv.eval()


    x = input("Enter the video path...")
    webcam = cv.VideoCapture(x)
    mvAvgx = []
    mvAvgy = []
    scale = 10
    margin = 200
    margin2 = 50
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    while True:
        ret, frame = webcam.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = detector(gray)
        for face in faces:
            # x, y = face.left(), face.top()
            # x1, y1 = face.right(), face.bottom()
            # cv2.rectangle(gray, (x, y), (x1, y1), (0, 255, 0), 2)
            landmarks = predictor(gray, face)
            # Gaze detection
            left_eye_region = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                        (landmarks.part(37).x, landmarks.part(37).y),
                                        (landmarks.part(38).x, landmarks.part(38).y),
                                        (landmarks.part(39).x, landmarks.part(39).y),
                                        (landmarks.part(40).x, landmarks.part(40).y),
                                        (landmarks.part(41).x, landmarks.part(41).y)], np.int32)
            # height, width, _ = frame.shape
            min_x = np.min(left_eye_region[:, 0])
            max_x = np.max(left_eye_region[:, 0])
            min_y = np.min(left_eye_region[:, 1])
            max_y = np.max(left_eye_region[:, 1])
            var = 8
            eye = gray[min_y - var: max_y + var, min_x - var: max_x + var]
            left_eye = cv.resize(eye, dsize=(100, 50))
            cv.imshow("frame", left_eye)
            top = max([max(x) for x in left_eye])
            left_eye = (torch.tensor([[left_eye]]).to(dtype=torch.float,
                                                      device=device)) / top

            x = ensembleX(left_eye)*1920
            y = fiv(left_eye).item()*1080-yshift
            pyautogui.FAILSAFE=False

            avx = sum(mvAvgx) / scale
            avy = sum(mvAvgy) / scale
            print(avx, avy)

            mvAvgx.append(x)
            mvAvgy.append(y)

            if len(mvAvgx) >= scale:
                if abs(avx - x) > margin and abs(avy - x) > margin:
                    mvAvgx = mvAvgx[5:]
                    mvAvgy = mvAvgy[5:]
                else:
                    if abs(avx - x) > margin2:
                        mvAvgx = mvAvgx[1:]
                    else:
                        mvAvgx.pop()

                    if abs(avy - y) > margin2:
                        mvAvgy = mvAvgy[1:]
                    else:
                        mvAvgy.pop()
                # else:
                #     mvAvgx = mvAvgx[1:]
                #     mvAvgy = mvAvgy[1:]
                # pyautogui.moveTo(720,450)
                pyautogui.moveTo(avx, avy)




        if cv.waitKey(1) & 0xFF == ord('q'):
           break

    return (mvAvgx, mvAvgy)








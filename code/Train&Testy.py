import numpy as np
import cv2 as cv
import os
import copy
import torch
import torch.nn as nn
import torchvision
import torch.functional as F
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        f2 = 4
        self.layer2 = nn.Sequ ential(
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


#ANOTHER_OPTION_TO_USE
#
# class ConvNet(torch.nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
#
#         f1 = 4
#         f2 = 16
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(f1, f2, kernel_size=5, stride=1, padding=2),
#
#             nn.ReLU(),
#             nn.BatchNorm2d(f2),
#             nn.MaxPool2d(kernel_size=2, stride=2))
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(1, f1, kernel_size=5, stride=1, padding=2),
#
#             nn.ReLU(),
#             nn.BatchNorm2d(f1),
#             nn.MaxPool2d(kernel_size=2, stride=2))
#         self.fc1 = nn.Linear(25 * 12 * f2, 400)
#         self.fc2 = nn.Linear(400, 60)
#         self.fc3 = nn.Linear(60, 1)
#
#
#
#     def forward(self,x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = x.reshape(x.size(0), -1)
#         x = self.fc1(x)
#         x = self.fc2(x)
#         x = self.fc3(x)
#
#         return x

def dataLoad(path, want = 0):
    nameList = os.listdir(path)

    try:
        nameList.remove(".DS_Store")
    except:
        pass
    totalHolder = []
    dims = [1440,900]

    for name in nameList:
        im = cv.cvtColor(cv.imread(path + "/" + name), cv.COLOR_BGR2GRAY)
        top = max([max(x) for x in im])
        totalHolder.append(((torch.tensor([[im]]).to(dtype=torch.float,device=device))/top,
                            torch.tensor([[int((name.split("."))[want])/dims[want]]]).to(dtype=torch.float,device=device)))

    #print(totalHolder)
    return totalHolder


def evaluateModel(model,testSet, sidelen = 1920):
    model.eval()
    err = 0
    for (im, label) in testSet:
        output = model(im)
        err += abs(output.item() - label.item())
        #print(output.item()-label.item())
    model.train()
    return (err/len(testSet)*sidelen)

def getError(model,testSet):
    model.eval()
    foeList = []
    for (im,label) in testSet:
        output = model(im)
        foeList.append(abs(output.item() - label.item()))

    model.train()
    return foeList




trainingSet = dataLoad("eyes_train")
test = dataLoad("eyes_test")


num_epochs = 10
bigTest = []
bigTrain = []

def trainModel():
    model = ConvNet().to(device)
    # model.cuda()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    bestModel = model
    bestScore = 10000
    testscores = []
    trainscores = []
    #print(trainingSet[0][1])
    model.train()
    for epoch in range(num_epochs):
        print(epoch)
        np.random.shuffle(trainingSet)

        for i,(im, label) in enumerate(trainingSet):

            output = model(im)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 798 == 0:
                testSc = evaluateModel(model,test,sidelen=1080)
                # = evaluateModel(model,test)
                trainSc = evaluateModel(model,trainingSet,sidelen=1080)
                #trainSc = evaluateModel(model,trainingSet)
                if testSc < bestScore:
                    bestModel = copy.deepcopy(model)
                    bestScore = testSc
                testscores.append(testSc)
                trainscores.append(trainSc)

                print(trainSc)
                print(testSc)
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                       .format(epoch+1, num_epochs, i+1, len(trainingSet), loss.item()))

    bigTest.append(testscores)
    bigTrain.append(trainscores)

    #finalScore = evaluateModel(bestModel,test)
    finalScore = evaluateModel(bestModel,test,sidelen=1080)
    print(finalScore)

    if finalScore < 150:
        torch.save(bestModel.state_dict(), "yModels/" + str(int(finalScore))+".plt")

    plt.title(str(int(finalScore)))
    plt.plot(testscores)
    plt.plot(trainscores)
    plt.show()

def train():
    #for i in range(6): (For continuously training different models. 6 different models are saved and you can get whichever is the best.
    trainModel()




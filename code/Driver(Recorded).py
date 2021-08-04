import traindata
import testdata
import Train&Testx
import Train&Testy
import VideoTracking
import matplotlib as plt


traindata.start()
testdata.start()
Train&Testx.train()
Train&Testy.train()
tup = VideoTracking.eyetrack()
print(tup[0])
plt.scatter(tup[0], tup[1])
plt.show()

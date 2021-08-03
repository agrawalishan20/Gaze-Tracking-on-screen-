# Gaze-Tracking-on-screen
Aim of the project is to track gaze using webcam and to analyze the data.

# Functionality
1. The tracker collects training and testing data of the user and asks the user to look at a moving cursor while keeping the **head still**  when looking at the so that the image    isn't blurred but often the model learns to handle the noise. I have taken 1600 training and 200 test images but it can varied by changing parameters in code/testdata.py and      code/traindata.py. A suggestion to collect more training data is that the training images can be inverted laterally. (and so will be the co-ordinates w.r.t vertical line          dividing the screen). I didn't implement this because this might mislead the model in tracking features.
2. There are two files code/Train&Testx.py and code/Train&Testy.py that train the x and y co-ordinate regressive models in 10 epochs with shuffling training data every epoch and      evaluates with the test set every half-epoch to get the best-model and best-model score respectively for both x and y models. The lower the score, better is the model.
3. Finally there are two files code/LiveTracking.py and code/VideoTracking.py that uses the same image-processing and loads both the x and y models to track live input coming from    webcam or a recorded video. The predicted points are stored in a list and finally a scatter-plot is made using matplotlib using the x and y co-ordinates.

# Requirements 
1. Requires all the necessary imports mentioned in the code. (dlib, copy, cv2, pyautogui, etc.) 
2. Uses shape_predictor_68_face_landmarks for features recognition.
3. **CUDA 11.1** enabled GPU is required with **cuDNN**
4. Use this link to get latest torch, torchvision & torchaudio for CUDA --> https://pytorch.org/

 

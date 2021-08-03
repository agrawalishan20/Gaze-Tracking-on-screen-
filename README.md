# Gaze-Tracking-on-screen-
Aim of the project is to track gaze using webcam and to analyze the data.

# Functionality
The tracker collects training and testing data of the user and asks the user to look at a moving cursor while keeping the **head still**  when looking at the so that the image isn't blurred but often the model learns to handle the noise. I have taken 1600 training and 200 test images but it can varied by changing parameters in /code/testdata.py and /code/traindata.py

# Requirements 
1. Requires all the necessary imports mentioned in the code. (dlib, copy, cv2, pyautogui, etc.) 
2. Uses shape_predictor_68_face_landmarks for features recognition.
3. **CUDA 11.1** enabled GPU is required with **cuDNN**
4. Use this link to get latest torch, torchvision & torchaudio for CUDA --> https://pytorch.org/

 

# RaspberryPi_Facial_Recognition_Camera
This is code for a Raspberry Pi 4 security camera with facial recognition and liveness detection. OpenCV is used for the facial recognition and TensorFlow is used for the liveness detection.

The folder named ```liveness-detection-opencv``` contains the code for the security camera with liveness detection and facial recognition.

How to run it:
1. Navigate to the folder 
2. Command:
    ```$ python liveness_demo.py --model liveness.model --le le.pickle \
	--detector face_detector```


The folder named ```opencv-face-recognition ``` contains the code for the security camera with just facial recognition.

How to run it:
1. Navigate to the folder
2. Command:
  ```
  $ python recognize_video.py --detector face_detection_model \
    --embedding-model openface_nn4.small2.v1.t7 \
    --recognizer output/recognizer.pickle \
    --le output/le.pickle
  ```

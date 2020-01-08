# USAGE
# python recognize_video.py --detector face_detection_model \
#   --embedding-model openface_nn4.small2.v1.t7 \
#   --recognizer output/recognizer.pickle \
#   --le output/le.pickle

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
from mail import sendEmail
import os
import glob

def recognize(frame, unknownImageCount, count):
    # load our serialized face detector from disk
    print("[INFO] loading face detector...")
    protoPath = os.path.sep.join(["face_detection_model", "deploy.prototxt"])
    modelPath = os.path.sep.join(["face_detection_model",
        "res10_300x300_ssd_iter_140000.caffemodel"])
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # load our serialized face embedding model from disk
    print("[INFO] loading face recognizer...")
    embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

    # load the actual face recognition model along with the label encoder
    recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read())
    le = pickle.loads(open("output/le.pickle", "rb").read())

    # initialize the video stream, then allow the camera sensor to warm up
    print("[INFO] starting video stream...")
   
    # loop over frames from the video file stream
    while True:
        # grab the frame from the threaded video stream

        # resize the frame to have a width of 600 pixels (while
        # maintaining the aspect ratio), and then grab the image
        # dimensions
        frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]

        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # apply OpenCV's deep learning-based face detector to localize
        # faces in the input image
        detector.setInput(imageBlob)
        detections = detector.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections
            if confidence > 0.5:
                # compute the (x, y)-coordinates of the bounding box for
                # the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # extract the face ROI
                face = frame[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                # ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue

                # construct a blob for the face ROI, then pass the blob
                # through our face embedding model to obtain the 128-d
                # quantification of the face
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                    (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                # perform classification to recognize the face
                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = le.classes_[j]
                frameCopy = frame
                # draw the bounding box of the face along with the
                # associated probability
                
                if(name == "Person1"): #green
                    text = "{}: {:.2f}%".format(name, proba * 100)
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                        (0, 255, 0), 2)
                    cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
                    updatedFrame = "frame.jpg"
                    cv2.imwrite(updatedFrame, frame)
                    if(count < 1):
                        print ("Sending email...")
                        sendEmail(text)
                        print ("Done")
                        
                        count = count + 1
                    print(name + " "  + str(proba *  100))
                if(name == "Person2"): #red
                    text = "{}: {:.2f}%".format(name, proba * 100)
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                        (0, 0, 255), 2)
                    cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                    updatedFrame = "frame.jpg"
                    cv2.imwrite(updatedFrame, frame)
                    if(count < 1):
                        print ("Sending email...")
                        sendEmail(text)
                        print ("Done")
                        
                        count = count + 1
                    print(name + " "  + str(proba *  100))
                if(name == "unknown"): #blue
                    text = "{}: {:.2f}%".format(name, proba * 100)
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                        (255, 0, 0), 2)
                    cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
                    updatedFrame = "frame.jpg"
                    cv2.imwrite(updatedFrame, frame)
                    if(count < 1):
                        print ("Sending email...")
                        sendEmail(text)
                        print ("Done")
                        
                        count = count + 1
                    print(name + " "  + str(proba *  100))
                    
                    if(((proba * 100) > 60) and (unknownImageCount < 15) and (name == "unknown")):
                        unknownFiles = len([f for f in os.listdir("dataset_recognize/unknown") if os.path.isfile(os.path.join("dataset_recognize/unknown", f))])
                        unknownFrame = str(unknownFiles) + ".jpg"
                        path = "dataset_recognize/unknown/"
                        print(unknownFrame)
                        cv2.imwrite(os.path.join(path, unknownFrame), frameCopy)
                        unknownImageCount = unknownImageCount + 1
                        #print("UNKNOWN COUNT: " + str(unknownImageCount))
        # update the FPS counter


        # show the output frame
        print("EMAIL COUNT: " + str(count))
        print("UNKNOWN COUNT: " + str(unknownImageCount))     
        #cv2.imshow("FrameR", frame)
        return frame, unknownImageCount, count
        # if t

import argparse
import os
import time
from datetime import datetime
import cv2
import face_recognition
import numpy as np
from imutils.video import VideoStream
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import sounddevice
from scipy.io.wavfile import write



path="ImageAttendence"
images = []
classNames = []
myList = os.listdir(path)
print(myList)


def atend():
    for cl in myList:
        curImg = cv2.imread('{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])

    print(classNames)

    def findEncoding(images):
        encodeList = []
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        return encodeList

    def markAttendence(name):
        with open('Attendence.csv', 'r+') as f:
            myDataList = f.readlines()
            nameList = []



            for line in myDataList:
                entry = line.split(',')
                nameList.append(entry[0])

            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime("%H:%M:%S")
                dateString = now.date()
                f.writelines('\n{name},{dtString},{dateString}')
                # tToSp()

    encodeListKnown = findEncoding(images)

    print('Encoding Complete')

    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            # print(faceDis)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                # print(name)
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                markAttendence(name)

        cv2.imshow('Webcam', img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        # return
    cv2.destroyAllWindows()
    cap.stop()

def dec2():
    def detect_and_predict_mask(frame, faceNet, maskNet):

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                     (104.0, 177.0, 123.0))

        faceNet.setInput(blob)
        detections = faceNet.forward()

        faces = []
        locs = []
        preds = []

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > args["confidence"]:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = np.expand_dims(face, axis=0)

                faces.append(face)
                locs.append((startX, startY, endX, endY))

        if len(faces) > 0:
            preds = maskNet.predict(faces)
        return (locs, preds)

    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--face", type=str, default="face_detector")
    ap.add_argument("-m", "--model", type=str, default="mask_detector.model")
    ap.add_argument("-c", "--confidence", type=float, default=0.5)
    args = vars(ap.parse_args())

    print("Loading FaceDetector...................")
    prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
    weightsPath = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    print("Loading FaceMask Detector...........")

    maskNet = load_model(args["model"])

    print("Starting Video Stream")

    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    while True:
        frame = vs.read()
        # frame = imutils.resize(frame, width=)

        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            if label=="Mask":
                # print("Mask")
                # cv2.destroyAllWindows()
                vs.stop()
                atend()
                # cv2.destroyWindow('frame')
                # break
            # elif label=="No Mask":
            #     # print("No")

            # label = "{}: {:.2f}%".format(label, max(mask,withoutMask)* 100 )

            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startY, startY), (endX, endY), color, 2)

        cv2.imshow("frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()

def takeSS():
    key = cv2.waitKey(1)
    webcam = cv2.VideoCapture(0)

    i = 0
    while True:
        i += 1
        try:
            check, frame = webcam.read()
            print(check)  # prints true as long as the webcam is running
            print(frame)  # prints matrix values of each framecd
            cv2.imshow("Capturing", frame)
            key = cv2.waitKey(1)
            if key == ord('s'):
                x = str(i)
                cv2.imwrite(filename="SSIMAGES/" + x + "saved_img.jpg", img=frame)
                webcam.release()
                img_new = cv2.imread(x + "saved_img.jpg", cv2.IMREAD_GRAYSCALE)
                # img_new = cv2.imshow("Captured Image", img_new)
                cv2.waitKey(1650)
                cv2.destroyAllWindows()
                print("Processing image...")
                img_ = cv2.imread(x + "saved_img.jpg", cv2.IMREAD_ANYCOLOR)
                print("Converting RGB image to grayscale...")
                # gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
                print("Converted RGB image to grayscale...")

                # img_resized = cv2.imwrite(filename='saved_img-final.jpg', img=gray)
                print("Image saved!")
                webcam.release()
                print("Camera off.")
                print("Program ended.")
                cv2.destroyAllWindows()
                break
            elif key == ord('q'):
                print("Turning off camera.")
                webcam.release()
                print("Camera off.")
                print("Program ended.")
                cv2.destroyAllWindows()
                break

        except(KeyboardInterrupt):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break



def record():
    pass
    fs = 44100
    print("Enter the Lecture duration (in seconds) - ")

    second = int(input())
    print("Recording ........ ")
    record_voice = sounddevice.rec(int(second * fs), samplerate=fs, channels=1)
    sounddevice.wait()
    print("Enter the name of the subject - ")
    name = input()
    print("Enter the lecture number (integer) - ")
    no = int(input())
    name = name + str(no)
    write("RECORD/" + name + ".wav", fs, record_voice)


import numpy as np
import cv2
import pickle

FaceCascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels = {}
with open("labels.pickle" , "rb") as f:
    labels = pickle.load(f)
    labels = {v:k for k,v in labels.items()}

cap = cv2.VideoCapture(0)

while True:
    #Capture frame by frame,
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FaceCascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for(x, y, w, h) in faces:
        #print(x, y, w, h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_col = frame[y:y+h, x:x+w]

        id_, conf = recognizer.predict(roi_gray)
        if conf>=45:
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (230, 208, 48)
            stroke = 2
            cv2.putText(frame, name, (x,y-15), font, 1, color, stroke, cv2.LINE_AA)

        imgItem = "my-face.png"
        cv2.imwrite(imgItem , roi_gray)

        color = (153, 0, 153)
        stroke = 3
        x2 = x + w
        y2 = y + h
        cv2.rectangle(frame, (x,y), (x2,y2), color, stroke)



    #Display the frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
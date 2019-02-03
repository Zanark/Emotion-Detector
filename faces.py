import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    #Capture frame by frame,
    ret, frame = cap.read()

    #Display the frame
    cv2.imshow('frame' , frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import os
import cv2
import numpy as np

from PIL import Image

FaceCascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR , "images")

c_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(IMAGE_DIR):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" " , "-").lower()
            print(label, path)
            if label in label_ids:
                pass
            else:
                label_ids[label] = c_id
                c_id += 1
            id_ = label_ids[label]
            print(label_ids)
            pil_image = Image.open(path)
            image_array = np.array(pil_image , "uint8")
            print(image_array)
            faces = FaceCascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

            for (x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

#print(y_labels)
#print(x_train)
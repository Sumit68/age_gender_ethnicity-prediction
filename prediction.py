import os
import cv2
import numpy as np
import pyforest
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image


def face_detection(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    faces = face_cascade.detectMultiScale(img, 1.1, 4)
    for (x, y, w, h) in faces:
    #cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        faces = img[y:y + h, x:x + w]
        #cv2.imshow("face",faces)
        cv2.imwrite('face.jpg', faces)


def img_preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (48, 48))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    img = img.reshape(1, 48, 48, 1)
    return img


def age_prediction(img):
    img = cv2.resize(img, (128, 128))
    #img = np.expand_dims(img, axis=0)
    img = img.reshape((1, 128, 128, 3))
    age_json_file = open('model.json', 'r')
    loaded_model_json = age_json_file.read()
    age_json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    age = int(loaded_model.predict(img)[0][0])
    return age



def gender_prediction(img):
    img = img_preprocessing(img)
    gender_json_file = open('gender_model.json', 'r')
    loaded_model_json = gender_json_file.read()
    gender_json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("gender_model.h5")
    gender = loaded_model.predict(img)
    print(gender[0][0])
    if gender[0][0] <= 0.5:
        gender = "Male"
    else:
        gender = "Female"
    return gender

def ethnicity_prediction(img):
    img = img_preprocessing(img)
    ethnicity_json_file = open('ethnicity_model.json', 'r')
    loaded_model_json = ethnicity_json_file.read()
    ethnicity_json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("ethnicity_model.h5")
    print(loaded_model.predict(img))
    ethnicity = np.argmax(loaded_model.predict(img))

    if ethnicity == 0:
        ethnicity = "White"
    elif ethnicity == 1:
        ethnicity = "Black"
    elif ethnicity == 2:
        ethnicity = "Asian"
    elif ethnicity == 3:
        ethnicity = "Indian"
    elif ethnicity == 4:
        ethnicity = "Cant Determine"
    return ethnicity






import face_recognition
import cv2
import pickle
import os

def load_model():
    model = pickle.load(open('celebrity_model.pkl', 'rb'))
    return model

def recognize_face(image_path, model):
    image = cv2.imread(image_path)
    faces = face_recognition.face_locations(image)
    encodings = face_recognition.face_encodings(image, faces)
    
    for encoding in encodings:
        predictions = model.predict([encoding])
        return predictions

    return None

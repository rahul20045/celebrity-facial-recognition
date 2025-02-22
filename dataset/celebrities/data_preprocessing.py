import cv2
import os
from face_recognition import face_locations, face_encodings

def preprocess_images(dataset_path):
    data = []
    labels = []
    
    celebrity_names = os.listdir(dataset_path)
    for label, celebrity_name in enumerate(celebrity_names):
        celebrity_folder = os.path.join(dataset_path, celebrity_name)
        for image_name in os.listdir(celebrity_folder):
            image_path = os.path.join(celebrity_folder, image_name)
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_locations(image)
            encodings = face_encodings(image, faces)
            
            for encoding in encodings:
                data.append(encoding)
                labels.append(label)
    
    return data, labels

from recognizer import load_model, recognize_face
from model import train_model
import cv2

def main():
    # Option 1: Train Model
    print("Training Model...")
    model = train_model()
    with open('celebrity_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Option 2: Recognize Celebrity in an Image
    print("Recognizing Celebrity...")
    image_path = 'test_image.jpg'
    model = load_model()
    celebrity = recognize_face(image_path, model)
    print(f'Identified Celebrity: {celebrity}')

if __name__ == '__main__':
    main()

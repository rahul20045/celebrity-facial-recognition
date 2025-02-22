from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from data_preprocessing import preprocess_images

def train_model():
    dataset_path = 'dataset/celebrities/'
    
    data, labels = preprocess_images(dataset_path)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy * 100}%')
    
    return model

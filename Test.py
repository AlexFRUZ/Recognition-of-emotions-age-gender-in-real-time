from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import pandas as pd
import os


TEST_DIR = 'train'
TRAIN_DIR = 'test'

model = load_model('emotiondetector.h5')

def createdataframe(directory):
    image_paths = []
    labels = []
    for label in os.listdir(directory):
        for imagename in os.listdir(os.path.join(directory, label)):
            image_paths.append(os.path.join(directory, label, imagename))
            labels.append(label)
        print(label, "завершено")
    return image_paths, labels

train = pd.DataFrame()
train['image'], train['label'] = createdataframe(TEST_DIR)

test = pd.DataFrame()
test['image'], test['label'] = createdataframe(TEST_DIR)

def extract_features(images):
    features = []
    for image in images:
        img = load_img(image, grayscale=True)
        img = np.array(img)
        features.append(img)
    features = np.array(features)
    features = features.reshape(len(features), 48, 48, 1)
    return features

test_features = extract_features(test['image'])
x_test = test_features / 255.0

y_pred = np.argmax(model.predict(x_test), axis=-1)

le = LabelEncoder()
le.fit(train['label'])
y_true_numeric = le.transform(test['label'])




accuracy = accuracy_score(y_true_numeric, y_pred)
precision = precision_score(y_true_numeric, y_pred, average='weighted')
recall = recall_score(y_true_numeric, y_pred, average='weighted')
f1 = f1_score(y_true_numeric, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_true_numeric, y_pred)


print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(conf_matrix)

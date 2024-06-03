from keras.utils import to_categorical
from keras.layers import Dropout, Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential
from keras.preprocessing.image import load_img, img_to_array
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder


num_classes = 2

TRAIN_DIR = 'dataset'


def create_dataframe(directory):
    image_path = []
    labels = []
    for label in os.listdir(directory):
        label_path = os.path.join(directory, label)
        if os.path.isdir(label_path):
            for image_name in os.listdir(label_path):
                image_path.append(os.path.join(label_path, image_name))
                labels.append(label)
    return image_path, labels

train = pd.DataFrame()
train['image'], train['label'] = create_dataframe(TRAIN_DIR)


def extract_features(images):
    features = []
    for image in tqdm(images):
        img = load_img(image, grayscale=True)
        img = img.resize((48, 48))
        img = img_to_array(img)
        features.append(img)
    features = np.array(features)
    features = features.reshape(len(features), 48, 48, 1)
    return features

train_features = extract_features(train['image'])

x_train = train_features / 255.0

le = LabelEncoder()
y_train = le.fit_transform(train['label'])
y_train = to_categorical(y_train, num_classes)

model = Sequential()

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=x_train, y=y_train, batch_size=128, epochs=40, validation_split=0.2)

model.save('gender.h5')

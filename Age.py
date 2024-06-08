import keras.callbacks
from keras.layers import Dense, Conv2D, AveragePooling2D, Flatten, Dropout
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.utils import to_categorical
from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import os
import numpy as np
import pandas as pd

num_class = 7

TRAIN_DIR = 'age_organized'
TEST_DIR = 'age_organize'


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

test = pd.DataFrame()
test['image'], test['label'] = create_dataframe(TEST_DIR)


def extract_features(images):
    features = []
    for image in tqdm(images):
        img = load_img(image, grayscale=True)
        img = img.resize((200, 200))
        img = img_to_array(img)
        features.append(img)
    features = np.array(features)
    features = features.reshape(len(features), 200, 200, 1)
    return features

train_features = extract_features(train['image'])
test_features = extract_features(test['image'])

x_train = train_features / 255.0
x_test = test_features / 255.0

le = LabelEncoder()
le.fit(train['label'])

y_train = le.fit_transform(train['label'])
y_test = le.fit_transform(test['label'])

y_train = to_categorical(y_train, num_class)
y_test = to_categorical(y_test, num_class)

# datagen = ImageDataGenerator(
#     rotation_range=10,
#     zoom_range=0.1,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     horizontal_flip=True
# )
#
# datagen.fit(x_train)

model = Sequential()

# model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(200, 200, 1)))
# model.add(AveragePooling2D(pool_size=(3, 3)))
# model.add(Dropout(0.2))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(200, 200, 1)))
model.add(AveragePooling2D(pool_size=(3, 3)))
model.add(Dropout(0.3))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(3, 3)))
model.add(Dropout(0.4))

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(3, 3)))
model.add(Dropout(0.2))

# model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
# model.add(AveragePooling2D(pool_size=(3, 3)))
# model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(num_class, activation='softmax'))


class StopAtAccuracy(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('val_accuracy') >= 0.9:
            print("\nReached validation accuracy of 0.9. Stopping training.")
            self.model.stop_training = True

stop_at_accuracy = StopAtAccuracy()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=x_train, y=y_train, batch_size=128, epochs=100, validation_data=(x_test, y_test), callbacks=[stop_at_accuracy])

model.save('age.h5')

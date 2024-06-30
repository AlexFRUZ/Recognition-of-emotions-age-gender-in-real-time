import os
import numpy as np
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
import cv2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array

dataset_path = r'D:\Train\test'

img_size = (48, 48)
num_classes = 7
epochs = 100

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=128,
    class_mode='categorical'
)

pretrained_model_path = r'D:\Train\emotion_model_final.h5'
pretrained_model = load_model(pretrained_model_path)

pretrained_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

def recognize_emotion(frame):
    resized_frame = cv2.resize(frame, (48, 48))

    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

    gray_frame = np.expand_dims(gray_frame, axis=-1)
    gray_frame = np.repeat(gray_frame, 3, axis=-1)

    gray_frame = gray_frame / 255.0  # Normalization to [0, 1]

    input_data = img_to_array(gray_frame)
    input_data = np.expand_dims(input_data, axis=0)

    emotion_probabilities = pretrained_model.predict(input_data)[0]
    emotion_label = np.argmax(emotion_probabilities)

    emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    detected_emotion = emotion_labels[emotion_label]

    print("Detected Emotion:", detected_emotion)

class EmotionRecognitionCallback(ModelCheckpoint):
    def on_epoch_end(self, epoch, logs=None):
        ret, frame = camera.read()
        if ret:
            recognize_emotion(frame)

camera = cv2.VideoCapture(0)

emotion_callback = EmotionRecognitionCallback(filepath=pretrained_model_path, monitor='loss', save_best_only=True)

pretrained_model.fit(train_generator, epochs=epochs, callbacks=[emotion_callback])

camera.release()

pretrained_model.save(pretrained_model_path)

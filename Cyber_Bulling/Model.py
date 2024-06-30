from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding, Dense, Dropout, Bidirectional, LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import os
import numpy as np
import nltk
from nltk.corpus import stopwords
import re

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

num_class = 5

TRAIN_DIR = r"D:\Train\Cyber_Bulling\Train"
TEST_DIR = r"D:\Train\Cyber_Bulling\Test"

def clean_text(text):
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

def load_texts_from_directory(directory):
    texts, labels = [], []
    for label, category in enumerate(['age', 'ethnicity', 'gender', 'not_cyberbullying', 'religion']): 
        category_dir = os.path.join(directory, category)
        for file_name in os.listdir(category_dir):
            with open(os.path.join(category_dir, file_name), 'r', encoding='utf-8') as file:
                text = file.read()
                text = clean_text(text)
                texts.append(text)
                labels.append(label)
    return texts, labels

train_texts, train_labels = load_texts_from_directory(TRAIN_DIR)
test_texts, test_labels = load_texts_from_directory(TEST_DIR)

max_words = 2000
maxlen = 100

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_texts + test_texts)
train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)
X_train = pad_sequences(train_sequences, maxlen=maxlen)
X_test = pad_sequences(test_sequences, maxlen=maxlen)
y_train = to_categorical(np.array(train_labels), num_classes=num_class)
y_test = to_categorical(np.array(test_labels), num_classes=num_class)

embedding_dim = 100

model = Sequential()

model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=maxlen))

model.add(Bidirectional(LSTM(100, activation='relu', return_sequences=True)))
model.add(Dropout(0.4))

model.add(Bidirectional(LSTM(128, activation='relu', return_sequences=True)))
model.add(Dropout(0.4))

model.add(Bidirectional(LSTM(156, activation='relu')))
model.add(Dropout(0.4))

model.add(Dense(128, activation='relu'))
model.add(Dense(num_class, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.save("cyberbullying_model.h5")
history = model.fit(X_train, y_train, epochs=7, batch_size=128, validation_data=(X_test, y_test))

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

accuracy = history.history['val_accuracy'][-1]
precision = precision_score(y_test_labels, y_pred, average='weighted')
recall = recall_score(y_test_labels, y_pred, average='weighted')
f1 = f1_score(y_test_labels, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test_labels, y_pred)

print("Metrics:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print("Confusion Matrix:")
print(conf_matrix)

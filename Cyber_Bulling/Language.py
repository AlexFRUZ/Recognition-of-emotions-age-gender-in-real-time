import os
import numpy as np
import tkinter as tk
from tkinter import scrolledtext, filedialog
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model, Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.callbacks import Callback

TRAIN_DIR = r'C:\Users\user1\OneDrive\Робочий стіл\Моя фігня\Сміття\Train'
TEST_DIR = r'C:\Users\user1\OneDrive\Робочий стіл\Моя фігня\Сміття\Train'

def load_texts_from_directory(directory):
    texts, labels = [], []
    for label, sentiment in enumerate(['Happy', 'Sad']):
        sentiment_dir = os.path.join(directory, sentiment)
        for filename in os.listdir(sentiment_dir):
            with open(os.path.join(sentiment_dir, filename), 'r', encoding='utf-8') as f:
                texts.append(f.read())
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
y_train = np.array(train_labels)
y_test = np.array(test_labels)

embedding_dim = 50

model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=maxlen))
model.add(Dropout(0.2))
model.add(LSTM(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='sigmoid'))

class StopAtAccuracy(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('val_accuracy') >= 0.94:
            print("\nReached validation accuracy of 0.94. Stopping training.")
            self.model.stop_training = True

stop_at_accuracy = StopAtAccuracy()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test), callbacks=stop_at_accuracy)

model_file_path = 'text_classification_model.h5'
model.save(model_file_path)

def load_test_document():
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if file_path:
        with open(file_path, 'r', encoding='utf-8') as file:
            input_text_widget.delete("1.0", tk.END)
            input_text_widget.insert(tk.END, file.read())

def classify_text():
    input_text = input_text_widget.get("1.0", tk.END).strip()
    if input_text:
        input_sequences = tokenizer.texts_to_sequences([input_text])
        input_sequence_padded = pad_sequences(input_sequences, maxlen=maxlen)
        probability = model.predict(input_sequence_padded)[0][0]
        result_label.config(text=f"Happy Probability: {probability:.4f}\nSad Probability: {1 - probability:.4f}")

def on_closing():
    model.save(model_file_path)
    root.destroy()

root = tk.Tk()
root.title("Text Classification App")

load_document_button = tk.Button(root, text="Load Test Document", command=load_test_document)
load_document_button.pack(pady=5)

input_text_widget = scrolledtext.ScrolledText(root, width=40, height=10, wrap=tk.WORD)
input_text_widget.pack(pady=10)

classify_button = tk.Button(root, text="Classify Text", command=classify_text)
classify_button.pack(pady=5)

result_label = tk.Label(root, text="")
result_label.pack(pady=10)

root.protocol("WM_DELETE_WINDOW", on_closing)

root.mainloop()

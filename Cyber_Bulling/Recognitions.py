import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QTextEdit, QMessageBox
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np

class TextClassificationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Text Classification App')
        self.setGeometry(100, 100, 600, 400)

        layout = QVBoxLayout()

        self.textEdit = QTextEdit()
        layout.addWidget(self.textEdit)

        self.classifyButton = QPushButton('Classify Text')
        self.classifyButton.clicked.connect(self.classifyText)
        layout.addWidget(self.classifyButton)

        self.resultLabel = QLabel()
        layout.addWidget(self.resultLabel)

        self.probabilityLabels = []
        for _ in range(5):  # Змінив на 5, оскільки у вас 5 класів
            label = QLabel()
            layout.addWidget(label)
            self.probabilityLabels.append(label)

        self.setLayout(layout)

        self.model = load_model(r"D:\Train\cyberbullying_model.h5")  # Завантажуємо модель

    def classifyText(self):
        text = self.textEdit.toPlainText()
        if text:
            max_words = 2000
            maxlen = 100

            # Токенізація тексту
            tokenizer = Tokenizer(num_words=max_words)
            tokenizer.fit_on_texts([text])
            sequence = tokenizer.texts_to_sequences([text])
            X = pad_sequences(sequence, maxlen=maxlen)

            # Розпізнавання
            y_pred_probs = self.model.predict(X)
            y_pred = np.argmax(y_pred_probs, axis=1)

            # Виведення результату
            categories = ['age', 'ethnicity', 'gender', 'not_cyberbullying', 'religion']
            predicted_category = categories[y_pred[0]]
            self.resultLabel.setText(f"Predicted Category: {predicted_category}")

            # Вивід ймовірностей для кожного класу
            for i, prob in enumerate(y_pred_probs[0]):
                self.probabilityLabels[i].setText(f"{categories[i]}: {prob:.4f}")

        else:
            QMessageBox.warning(self, 'Warning', 'Please enter some text.')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TextClassificationApp()
    window.show()
    sys.exit(app.exec_())

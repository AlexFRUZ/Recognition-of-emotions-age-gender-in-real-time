import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import numpy as np

age_ranges = ['15-20', '25-32', '38-43', '4-6', '48-53', '60+', '8-13']


class AgeClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.model = load_model('age.h5')

    def initUI(self):
        self.setWindowTitle('Age Classifier')
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.btn_load = QPushButton('Load Image', self)
        self.btn_load.clicked.connect(self.load_image)
        self.layout.addWidget(self.btn_load)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)

        self.result_label = QLabel(self)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.result_label)

    def load_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.xpm *.jpg *.jpeg)", options=options)
        if file_path:
            self.display_image(file_path)
            self.predict_age(file_path)

    def display_image(self, file_path):
        pixmap = QPixmap(file_path)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))

    def predict_age(self, file_path):
        img = load_img(file_path, color_mode='grayscale', target_size=(48, 48))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        predictions = self.model.predict(img_array)
        predicted_age_index = np.argmax(predictions)
        predicted_age_range = age_ranges[predicted_age_index]
        confidence = predictions[0][predicted_age_index] * 100

        self.result_label.setText(f'Predicted Age Range: {predicted_age_range} ({confidence:.2f}%)')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AgeClassifierApp()
    ex.show()
    sys.exit(app.exec_())

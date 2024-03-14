import cv2
from keras.models import model_from_json
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QSplitter
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

class EmotionDetectorApp(QWidget):
    def __init__(self):
        super().__init__()

        self.timer = None
        self.canvas = None
        self.ax = None
        self.figure = None
        self.emotion_counts = None
        self.statistics_label = None
        self.btn_stat = None
        self.webcam = None
        self.labels = None
        self.haar_file = None
        self.face_cascade = None
        self.model = None
        self.model_json = None
        self.json_file = None
        self.initUI()

    def initUI(self):
        self.json_file = open("emotiondetector.json", "r")
        self.model_json = self.json_file.read()
        self.json_file.close()
        self.model = model_from_json(self.model_json)

        self.model.load_weights("emotiondetector.h5")
        self.haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(self.haar_file)

        self.webcam = cv2.VideoCapture(0)
        self.labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

        self.btn_stat = QPushButton('Show Emotion Statistics', self)
        self.btn_stat.clicked.connect(self.show_statistics)

        self.statistics_label = QLabel(self)
        self.emotion_counts = {label: 0 for label in self.labels.values()}  # Initialize counts for each emotion

        # Matplotlib setup
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.ax.bar(self.labels.values(), self.emotion_counts.values())
        self.ax.set_ylabel('Count')
        self.ax.set_title('Emotion Statistics')

        self.splitter = QSplitter(Qt.Vertical)
        self.splitter.addWidget(self.statistics_label)
        self.splitter.addWidget(self.canvas)

        layout = QVBoxLayout()
        layout.addWidget(self.btn_stat)
        layout.addWidget(self.splitter)

        self.setLayout(layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.showMaximized()

    def update_frame(self):
        i, im = self.webcam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(im, 1.3, 5)

        try:
            for (p, q, r, s) in faces:
                image = gray[q:q + s, p:p + r]
                cv2.rectangle(im, (p, q), (p + r, q + s), (255, 0, 0), 2)
                image = cv2.resize(image, (48, 48))
                img = extract_features(image)
                pred = self.model.predict(img)
                prediction_label = self.labels[pred.argmax()]

                cv2.putText(im, '% s' % prediction_label, (p - 10, q - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,
                            (0, 0, 255))

                self.emotion_counts[prediction_label] += 1

            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            h, w, ch = im.shape
            bytes_per_line = ch * w
            q_image = QImage(im.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.statistics_label.setPixmap(pixmap)

            cv2.waitKey(27)
        except cv2.error:
            pass

    def show_statistics(self):
        self.ax.clear()
        self.ax.bar(self.labels.values(), self.emotion_counts.values())
        self.ax.set_ylabel('Count')
        self.ax.set_title('Emotion Statistics')
        self.canvas.draw()


if __name__ == '__main__':
    app = QApplication([])
    window = EmotionDetectorApp()
    app.exec_()

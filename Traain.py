import cv2
from keras.models import model_from_json
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QMessageBox, QTextEdit, QGridLayout
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0


class PositiveEmotionWindow(QWidget):
    def __init__(self, emotion_counts):
        super().__init__()
        self.setWindowTitle('Emotion Levels')
        self.resize(600, 500)

        self.figure, (self.ax_positive, self.ax_neutral, self.ax_negative) = plt.subplots(3, 1, figsize=(8, 6))
        self.ax_positive.set_title('Positive Emotions')
        self.ax_neutral.set_title('Neutral Emotions')
        self.ax_negative.set_title('Negative Emotions')

        self.emotion_line_positive, = self.ax_positive.plot([], [], 'b-', label='Emotion Levels')
        self.emotion_line_neutral, = self.ax_neutral.plot([], [], 'g-', label='Emotion Levels')
        self.emotion_line_negative, = self.ax_negative.plot([], [], 'r-', label='Emotion Levels')

        self.ax_positive.set_xticks(range(7))
        self.ax_neutral.set_xticks(range(7))
        self.ax_negative.set_xticks(range(7))

        self.ax_positive.set_ylabel('Emotion Level')
        self.ax_neutral.set_ylabel('Emotion Level')
        self.ax_negative.set_ylabel('Emotion Level')

        self.ax_positive.legend()
        self.ax_neutral.legend()
        self.ax_negative.legend()

        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.x_data = {'positive': [], 'neutral': [], 'negative': []}
        self.y_data = {'positive': [], 'neutral': [], 'negative': []}

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(1000)

        self.emotion_counts = emotion_counts

    def update_plot(self):
        positive_emotions = ['happy', 'surprise']
        neutral_emotions = ['neutral']
        negative_emotions = ['angry', 'disgust', 'fear', 'sad']

        total_positive_emotion = sum([self.emotion_counts[emotion] for emotion in positive_emotions])
        total_neutral_emotion = sum([self.emotion_counts[emotion] for emotion in neutral_emotions])
        total_negative_emotion = sum([self.emotion_counts[emotion] for emotion in negative_emotions])

        self.x_data['positive'].append(len(self.x_data['positive']) + 1)
        self.x_data['neutral'].append(len(self.x_data['neutral']) + 1)
        self.x_data['negative'].append(len(self.x_data['negative']) + 1)

        self.y_data['positive'].append(total_positive_emotion)
        self.y_data['neutral'].append(total_neutral_emotion)
        self.y_data['negative'].append(total_negative_emotion)

        self.emotion_line_positive.set_data(self.x_data['positive'], self.y_data['positive'])
        self.emotion_line_neutral.set_data(self.x_data['neutral'], self.y_data['neutral'])
        self.emotion_line_negative.set_data(self.x_data['negative'], self.y_data['negative'])

        for ax in [self.ax_positive, self.ax_neutral, self.ax_negative]:
            ax.relim()
            ax.autoscale_view(True, True, True)

        self.canvas.draw()


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
        self.btn_positive = None
        self.webcam = None
        self.labels = None
        self.haar_file = None
        self.face_cascade = None
        self.model = None
        self.model_json = None
        self.json_file = None
        self.positive_window = None
        self.training_info = None
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

        self.btn_positive = QPushButton('Show Positive Emotions', self)
        self.btn_positive.clicked.connect(self.show_positive_emotions)

        self.statistics_label = QLabel(self)
        self.emotion_counts = {label: 0 for label in self.labels.values()}

        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.ax.bar(self.labels.values(), self.emotion_counts.values())
        self.ax.set_ylabel('Count')
        self.ax.set_title('Emotion Statistics')

        try:
            with open("ML.txt", "r") as file:
                training_info = file.read()
            self.training_info = training_info
        except FileNotFoundError:
            self.training_info = "File not found."
        layout = QVBoxLayout()
        grid_layout = QGridLayout()

        grid_layout.addWidget(self.btn_stat, 0, 0)
        grid_layout.addWidget(self.btn_positive, 0, 1)
        grid_layout.addWidget(self.statistics_label, 1, 0, 1, 2)
        grid_layout.addWidget(self.canvas, 4, 0, 1, 2)

        self.text_edit = QTextEdit()
        self.text_edit.setText(self.training_info)

        grid_layout.addWidget(self.text_edit, 0, 12, 0, 10)

        layout.addLayout(grid_layout)
        layout.setAlignment(Qt.AlignTop)

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

    def show_positive_emotions(self):
        self.positive_window = PositiveEmotionWindow(self.emotion_counts)
        self.positive_window.show()

    def show_statistics(self):
        self.ax.clear()
        self.ax.bar(self.labels.values(), self.emotion_counts.values())
        self.ax.set_ylabel('Count')
        self.ax.set_title('Emotion Statistics')
        self.canvas.draw()
        dominant_emotion = max(self.emotion_counts, key=self.emotion_counts.get)
        if dominant_emotion == 'angry':
            message = "You seem to be angry."
        elif dominant_emotion == 'disgust':
            message = "You seem to be disgusted."
        elif dominant_emotion == 'fear':
            message = "You seem to be afraid."
        elif dominant_emotion == 'happy':
            message = "You seem to be happy."
        elif dominant_emotion == 'neutral':
            message = "You seem to be neutral."
        elif dominant_emotion == 'sad':
            message = "You seem to be sad."
        elif dominant_emotion == 'surprise':
            message = "You seem to be surprised."
        else:
            message = "Unable to determine your emotion."

        QMessageBox.information(self, "Dominant Emotion", message)

if __name__ == '__main__':
    app = QApplication([])
    window = EmotionDetectorApp()
    window.show()
    app.exec_()

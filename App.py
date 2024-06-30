import sys
import cv2
import numpy as np
from keras.models import model_from_json, load_model
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QTextEdit, QGridLayout
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
        self.timer.start(100)

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
        self.emotion_model = None
        self.gender_model = None
        self.age_model = None
        self.model_json = None
        self.json_file = None
        self.positive_window = None
        self.training_info = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Recognition App")

        self.json_file = open("emotiondetector.json", "r")
        self.model_json = self.json_file.read()
        self.json_file.close()
        self.emotion_model = model_from_json(self.model_json)
       
        self.emotion_model.load_weights("emotiondetector.h5")

        self.gender_model = load_model("gender.h5")

        self.age_model = load_model("age.h5")

        self.haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(self.haar_file)

        self.webcam = cv2.VideoCapture(0)
        self.labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

        self.age_ranges = ['15-20', '25-32', '38-43', '4-6', '48-53', '60+', '8-13']

        self.btn_stat = QPushButton('Show Emotion Statistics', self)
        self.btn_stat.clicked.connect(self.show_statistics)

        self.btn_positive = QPushButton('Show Emotions', self)
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
        self.text_edit.setReadOnly(True)
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

                pred_emotion = self.emotion_model.predict(img)
                prediction_label_emotion = self.labels[pred_emotion.argmax()]

                pred_gender = self.gender_model.predict(img)
                gender_label = "Male" if pred_gender[0][0] > 0.5 else "Female"

                pred_age = self.age_model.predict(img)
                age_range_index = pred_age.argmax()
                age_range_label = self.age_ranges[age_range_index]

                cv2.putText(im, f'{gender_label}: {prediction_label_emotion} ({age_range_label})', (p - 10, q - 10),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255))

                self.emotion_counts[prediction_label_emotion] += 1

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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = EmotionDetectorApp()
    window.show()
    sys.exit(app.exec_())

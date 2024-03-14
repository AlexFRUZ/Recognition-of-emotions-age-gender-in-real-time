import sys

from PyQt5 import QtWidgets as qw

class Windows(qw.QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Square a rectangle")
        self.setGeometry(840, 380, 100, 150)

        self.layout = qw.QVBoxLayout()

        self.label = qw.QLabel("Enter two numbers:")
        self.layout.addWidget(self.label)

        self.input1 = qw.QLineEdit()
        self.layout.addWidget(self.input1)

        self.input2 = qw.QLineEdit()
        self.layout.addWidget(self.input2)

        self.caluclate_button = qw.QPushButton("Calculate")
        self.layout.addWidget(self.caluclate_button)
        self.caluclate_button.clicked.connect(self.square_rectangle)

        self.setLayout(self.layout)

    def square_rectangle(self):
        try:
            a = float(self.input1.text())
            b = float(self.input2.text())
            result = a * b
            qw.QMessageBox.information(self, "Result", f"Result:{result}")
        except ValueError as ve:
            qw.QMessageBox.warning(self, "Error", f"{ve}, Please enter valid numbers")

if __name__ == "__main__":
    app = qw.QApplication(sys.argv)
    window = Windows()
    window.show()
    sys.exit(app.exec_())

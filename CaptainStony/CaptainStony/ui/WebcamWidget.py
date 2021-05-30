import sys
from PyQt5 import QtGui, QtCore
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QWidget, QPushButton, QVBoxLayout, QLabel
import cv2

class WebcamWidget(QWidget):
    def __init__(self, *args):
        super(QWidget, self).__init__()

        self.fps = 24
        self.cap = cv2.VideoCapture(*args)

        self.video_frame = QLabel()
        lay = QVBoxLayout()
        #lay.setMargin(0)
        lay.addWidget(self.video_frame)
        self.setLayout(lay)

        # ------ Modification ------ #
        self.isCapturing = False
        self.ith_frame = 1
        # ------ Modification ------ #

    def setFPS(self, fps):
        self.fps = fps

    def nextFrameSlot(self):
        ret, frame = self.cap.read()

        # ------ Modification ------ #
        # Save images if isCapturing
        if self.isCapturing:
            cv2.imwrite('img_%05d.jpg'%self.ith_frame, frame)
            self.ith_frame += 1
            self.isCapturing = False
        # ------ Modification ------ #

        # My webcam yields frames in BGR format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        img = img.scaled(320, 240)
        pix = QPixmap.fromImage(img)
        self.video_frame.setPixmap(pix)

    def start(self):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.nextFrameSlot)
        self.timer.start(1000./self.fps)

    def stop(self):
        self.timer.stop()

    def capture(self):
        if not self.isCapturing:
            self.isCapturing = True
        else:
            self.isCapturing = False

    def deleteLater(self):
        self.cap.release()
        super(QWidget, self).deleteLater()


class WebcamControlWindow(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.capture = None

        self.start_button = QPushButton('Initialize camera')
        self.start_button.clicked.connect(self.startCapture)
        self.quit_button = QPushButton('End')
        self.quit_button.clicked.connect(self.endCapture)
        self.end_button = QPushButton('Stop')

        self.capture_button = QPushButton('Shoot!')
        self.capture_button.clicked.connect(self.saveCapture)

        vbox = QVBoxLayout(self)
        vbox.addWidget(self.start_button)
        vbox.addWidget(self.end_button)
        vbox.addWidget(self.quit_button)

        vbox.addWidget(self.capture_button)

        self.setLayout(vbox)
        self.setWindowTitle('Control Panel')
        #self.setGeometry(400,400,400,400)
        self.show()

    def startCapture(self):
        if not self.capture:
            self.capture = WebcamWidget(0)
            self.end_button.clicked.connect(self.capture.stop)
            # self.capture.setFPS(1)
            self.capture.setParent(self)
            #self.capture.setWindowFlags(QtCore.Qt.Tool)
            self.layout().addWidget(self.capture)

        self.capture.start()
        self.update()
        #self.capture.show()

    def endCapture(self):
        self.capture.deleteLater()
        self.capture = None

    def saveCapture(self):
        if self.capture:
            self.capture.capture()



if __name__ == '__main__':

    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = WebcamControlWindow()
    sys.exit(app.exec_())
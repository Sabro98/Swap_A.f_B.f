from PyQt5.QtWidgets import *
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5 import QtCore
import threading
import sys
import os
import cv2
import numpy as np
from MPProcess import Swapper
import mediapipe as mp
from time import sleep


class HelpDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HELP!!")
        self.setGeometry(400, 150, 700, 750)
        self.label = QLabel()
        self.label.setText(
            '''
            이 프로그램은 나의 얼굴을 원하는 얼굴로 바꿔줍니다.\n\n
            
            먼저, 원하는 얼굴 영상을 "사진 불러오기" 버튼을 통해 불러와주세요.\n
            이때, 얼굴이 잘 드러난 영상이면 최대한 좋습니다. 샘플 파일(sample.jpg)을 확인해 주세요.\n
            "사진 불러오기"를 성공했다면, 오른쪽에 불러온 사진과 face mesh가 보일 것입니다.\n
            이는 나의 얼굴을 face mesh 부분으로 대체한다는 것을 의미합니다.\n\n
            
            그다음으로, 카메라를 켜기 위해 "비디오 켜기" 버튼을 눌러주세요.\n
            잠깐! 누르기 전에 마음의 준비를 하고 눌러주세요. 얼굴이 바뀌어 있을 수도 있습니다.\n
            
            바뀐 얼굴이 재미있나요? 그렇다면 이 순간을 보관해 보세요!\n
            "사진 저장하기"버튼을 누르면 현재 순간을 사진으로 저장해 줍니다.\n
            저장된 사진은 save.jpg이므로 잊지 말고 챙겨주세요! 두 개 이상은 저장이 안 됩니다.\n\n
            
            또 다른 얼굴로 바꾸고 싶나요? 그렇다면 "사진 불러오기"를 통해 바꾸면 됩니다.\n
            여러 유명인의 얼굴 영상이 images 폴더에 들어있습니다. 참고해 주세요!\n
            참고로, 사람 얼굴이 아닌 것을 입력 또는 2명 이상을 입력할 경우 정상적으로 동작하지 않을 수 있습니다!\n
            그럼, 재미있게 즐겨주세요!\n
            
            !주의! images/sample9.jpg 영상은 특별히 프로그램 초기화에 사용되니 주의해주세요!\n
            '''
        )
        self.quit_button = QPushButton("OK")
        self.quit_button.clicked.connect(self.close_function)

        wid = QtWidgets.QWidget(self)
        vbox = QVBoxLayout()
        vbox.addWidget(self.label)
        vbox.addWidget(self.quit_button)
        wid.setLayout(vbox)

    def close_function(self):
        self.close()

    def show_modal(self):
        return super().exec_()


class Video(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Swap face")
        self.setGeometry(600, 200, 1000, 600)
        self.frame = None
        self.swapper = None
        self.templateImage = None
        self.videoON = False
        self.currentImage = None

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)

        # set buttons
        self.video_on_button = QPushButton("비디오 켜기", self)
        self.video_off_button = QPushButton("비디오 끄기", self)
        self.save_image_button = QPushButton("사진 저장", self)
        self.load_button = QPushButton("사진 불러오기", self)
        self.help_button = QPushButton("사용법")
        self.quit_button = QPushButton("나기기", self)

        self.video_on_button.clicked.connect(self.video_on_function)
        self.video_off_button.clicked.connect(self.video_off_function)
        self.save_image_button.clicked.connect(self.save_image_function)
        self.quit_button.clicked.connect(self.quit_function)
        self.load_button.clicked.connect(self.load_function)
        self.help_button.clicked.connect(self.help_function)

        self.video_off_button.setDisabled(True)
        self.save_image_button.setDisabled(True)

        self.infoLabel = QLabel()
        self.infoLabel.setAlignment(QtCore.Qt.AlignCenter)

        # set canvas
        self.videoCanvas = QLabel()
        init_template_image = np.ones_like(cv2.imread('./images/sample9.jpg')) * 255
        self.init_video_image = np.zeros_like(init_template_image)
        self.currentImage = self.init_video_image
        self.set_image(self.videoCanvas, self.init_video_image, size=(640, 360))
        self.templateCanvas = QLabel()
        self.set_image(self.templateCanvas, init_template_image, size=(320, 320))

        # set layout
        wid = QtWidgets.QWidget(self)
        self.setCentralWidget(wid)

        hbox = QHBoxLayout()
        hbox.addWidget(self.videoCanvas)
        hbox.addWidget(self.templateCanvas)

        hbox1 = QHBoxLayout()
        hbox1.addWidget(self.video_on_button)
        hbox1.addWidget(self.video_off_button)

        hbox2 = QHBoxLayout()
        hbox2.addWidget(self.save_image_button)
        hbox2.addWidget(self.load_button)

        hbox3 = QHBoxLayout()
        hbox3.addWidget(self.help_button)
        hbox3.addWidget(self.quit_button)

        vbox = QVBoxLayout()
        vbox.addLayout(hbox)
        vbox.addWidget(self.infoLabel)
        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)
        vbox.addLayout(hbox3)

        wid.setLayout(vbox)

    def info(self, text: str):
        self.infoLabel.setText(text)

    def help_function(self):
        dialog = HelpDialog()
        dialog.exec_()

    def save_image_function(self):
        cv2.imwrite('./save.jpg', cv2.flip(self.currentImage, 1))
        self.info("저장 완료! save.jpg를 확인해주세요!")

    def load_function(self):
        prev_start = self.videoON
        if prev_start:
            self.stop_video()
        want_type = "Images (*.png *.jpg *.jpeg *tif)"
        file_path, file_type = QtWidgets.QFileDialog.getOpenFileName(self, "Open file", os.getcwd(),
                                                                     self.tr(want_type))
        if file_path == '' or file_type != want_type:
            return

        self.templateImage = cv2.imread(file_path)
        success, mesh_img = self.mesh_face(self.templateImage)
        if success:
            self.set_image(self.templateCanvas, mesh_img, size=(320, 320))
            self.swapper = Swapper(file_path)
        else:
            self.set_image(self.templateCanvas, self.templateImage, size=(320, 320))
            self.swapper = None

        if prev_start:
            self.start_video()

        self.info('영상 불러오기 성공!')

    def set_image(self, canvas: QLabel, input_img: np.ndarray, size: tuple = None):
        img = input_img.copy()

        # resize image
        if size is not None:
            img = cv2.resize(img, dsize=size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = img.shape
        q_img = QtGui.QImage(img.data, w, h, w * c, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(q_img)
        canvas.setPixmap(pixmap)

    def video_thread(self):
        while self.videoON:
            ret, self.frame = self.cap.read()
            if not ret:
                break
            img = self.frame.copy()
            try:
                if self.swapper is not None:
                    img = self.swapper.processing(self.frame)
                    self.currentImage = img
            except IndexError:
                break
            except ValueError:
                break
            self.set_image(self.videoCanvas, cv2.flip(img, 1), size=(640, 360))
        self.set_image(self.videoCanvas, self.init_video_image, size=(640, 360))

    def video_off_function(self):
        self.stop_video()
        self.info("비디오 끄기 성공!")

    def stop_video(self):
        self.videoON = False
        self.video_off_button.setDisabled(True)
        self.video_on_button.setEnabled(True)
        self.save_image_button.setDisabled(True)

    def start_video(self):
        self.videoON = True
        self.video_on_button.setDisabled(True)
        self.video_off_button.setEnabled(True)
        self.save_image_button.setEnabled(True)
        t = threading.Thread(target=self.video_thread)
        t.start()

    def video_on_function(self):
        if not self.cap.isOpened():
            self.close()
        self.start_video()
        self.info("비디오 켜기 성공!")

    def mesh_face(self, input_img: np.ndarray) -> (bool, np.ndarray):
        mp_mesh = mp.solutions.face_mesh
        mp_drawing = mp.solutions.drawing_utils
        mp_styles = mp.solutions.drawing_styles
        out_img = input_img.copy()
        mesh = mp_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5,
                                min_tracking_confidence=0.5)

        res = mesh.process(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB))
        success = False
        if res.multi_face_landmarks:
            success = True
            for landmarks in res.multi_face_landmarks:
                mp_drawing.draw_landmarks(image=out_img, landmark_list=landmarks,
                                          connections=mp_mesh.FACEMESH_CONTOURS,
                                          landmark_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=1),
                                          connection_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=1))
        return success, out_img

    def quit_function(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Video()
    win.show()
    sys.exit(app.exec_())

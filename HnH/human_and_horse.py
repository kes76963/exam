import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QPixmap
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

form_window = uic.loadUiType('./mainWidget2.ui')[0] # 아까 만든 ui를 클래스 형태로 로드
FROM_CLASS_Loading = uic.loadUiType("load2.ui")[0]


class Exam(QWidget, form_window):
    def __init__(self):
        super().__init__()
        self.path = None
        self.setupUi(self)
        #self.btn_exit.clicked.connect(QCoreApplication.instance().quit) # 버튼을 클릭하면, 종료를 하는 함수에 연결
                                                                        # quit = 윈도우 종료 함수. quit은 즉시 종료, 윈도우 x를 통한 닫기는 정리하고 종료.
        self.model = load_model('../models/human_and_horse_classification.h5')

        self.btn_select.clicked.connect(self.loading)
        self.btn_select.clicked.connect(self.predict_image) # 함수 뒤에 ()를 넣으면 호출되므로 쓰지 않음. 함수 연결만 하면 됨.

    def loading(self):
        # 로딩중일때 다시 클릭하는 경우
        try:
            self.loading
            self.loading.deleteLater()

        # 처음 클릭하는 경우
        except:
            self.loading = loading(self)
            self.loading.deleteLater()

    def predict_image(self):    # 예측하는 함수
        self.path = QFileDialog.getOpenFileName(                # 파일 경로 읽기. 윈도우면 윈도우 열기 형태, 맥이면 맥 열기 형태, 리눅스면 리눅스 열기 형태 창 뜸
            self,
            "Open file", '../datasets/train2',  # 가장 처음에 오픈하는 위치
            "Image Files(*.png);;All Files(*.*)"   # 보고싶은 파일 항목 볼수 있음 ; 2개로 구분 지음
        )
        print(self.path)
        if self.path[0]:
            pixmap = QPixmap(self.path[0])    # 라벨에 이미지 출력하려면 QPixmap 사용해야 함. QPixmap(경로 지정) // self.path[0]에는 경로가 저장되어 있음
            self.lbl_image.setPixmap(pixmap)

            try:
                img = Image.open(self.path[0])
                img = img.convert('RGB')
                img = img.resize((64, 64))
                data = np.asarray(img)
                data = data / 255
                data = data.reshape(1, 64, 64, 3)
            except:
                print('error')
            predict_value = self.model.predict(data)
            if predict_value > 0.5:
                self.lbl_predict.setText('이 이미지는 ' +
                     str((predict_value[0][0]*100).round()) +'% 확률로 Human입니다.')

            else:
                self.lbl_predict.setText('이 이미지는 ' +
                     str(((1 - predict_value[0][0])*100).round()) +'% 확률로 Horse입니다.')


class loading(QWidget, FROM_CLASS_Loading):

    def __init__(self, parent):
        super(loading, self).__init__(parent)
        self.setupUi(self)
        self.center()
        self.show()

        # 동적 이미지 추가
        self.movie = QMovie('loading.gif', QByteArray(), self)
        self.movie.setCacheMode(QMovie.CacheAll)
        # QLabel에 동적 이미지 삽입
        self.label.setMovie(self.movie)
        self.movie.start()
        # 윈도우 해더 숨기기
        self.setWindowFlags(Qt.FramelessWindowHint)

    # 위젯 정중앙 위치
    def center(self):
        size = self.size()
        ph = self.parent().geometry().height()
        pw = self.parent().geometry().width()
        self.move(int(pw / 2 - size.width() / 2), int(ph / 2 - size.height() / 2))

app = QApplication(sys.argv)
mainWindow = Exam()
mainWindow.show()
sys.exit(app.exec_())

import cv2
import numpy as np
from PIL import Image
from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap, QImage

from untitled import Ui_MainWindow
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel, QMessageBox
from predict_copy import predict
from yolo import YOLO
yolo=YOLO()
crop=False
count=False
class CamShow(QMainWindow,Ui_MainWindow):
    def __init__(self,parent=None):
        super(CamShow,self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("yolov3目标检测与定位系统")
        self.image=None
        self.openfile_name_image = ''



        # self.label = QLabel(self)
        # self.label.setText("   待检测图片")
        # self.label.setFixedSize(700, 500)
        # self.label.move(110, 80)


        self.pushButton.clicked.connect(self.loadimg)
        self.pushButton_2.clicked.connect(self.detect)
        self.pushButton_3.clicked.connect(self.close)

    def closeEvent(self, event):
        result = QMessageBox.question(self, "警告：", "退出将结束运行，是否继续！！",
                                      QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if result == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()
    def loadimg(self):
        # self.num+=1
        # self.label_2.setText(str(self.num))
        # fname,_=QFileDialog.getOpenFileNames(self, "选择图片", "./img", "Images (*.jpg *.jpeg *.png)")
        temp, _ = QFileDialog.getOpenFileName(self, "选择照片文件", r"./img/")
        if temp is not None:
            self.openfile_name_image = temp
        #     读取选择的图片
        self.image = cv2.imread(self.openfile_name_image)
        # print(self.openfile_name_image)
        # 将路径中的图片读取之后放在self.label2
        self.label_1.setPixmap(QPixmap(str(self.openfile_name_image)))
        self.label_1.setScaledContents(True)
        # 读取收缩放至(400, 300)
        self.label_1.setMaximumSize(550, 400)
        self.label_1.setScaledContents(True)

    def first_detect(self,path):
        try:
            image = Image.open(path)
        except:
            print('Open Error! Try again!')
        else:
            ##这里是我模型检测函数，替换成自己的即可，这个函数返回的就是检测好的图片，然后保存在本地的同级目录下的img/result

            r_image ,str1= yolo.detect_image(image, crop=crop, count=count)
            r_image.save('img_out/xin/' + path.split('/')[-1])
            f = open(r"./test.txt", 'r')
            s = f.read()

            self.label_2.setText(s)
    def detect(self):
        if self.image is None:

            print('未找到图片')
        elif self.image is not None:
            self.first_detect(self.openfile_name_image)
            img = cv2.imread('img_out/xin/' + self.openfile_name_image.split('/')[-1])
            img = cv2.resize(img, (550, 400), interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # cv2.imshow('test', img)
            # cv2.waitKey(20)
            # 将图片放在标签self.label3中
            a = QImage(img.data, img.shape[1], img.shape[0], img.shape[1] * 3, QImage.Format_RGB888)
            self.label_3.setPixmap(QPixmap.fromImage(a))
        pass





if __name__ == '__main__':
    app=QApplication(sys.argv)
    ui=CamShow()
    ui.show()
    sys.exit(app.exec_())

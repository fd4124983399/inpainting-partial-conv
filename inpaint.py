"""
PyQt App that leverages completed model for image inpainting
"""

import sys
import os
import random
import torch
import argparse

from PIL import Image
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from torchvision.utils import make_grid
from torchvision.utils import save_image
from torchvision import transforms

from partial_conv_net import PartialConvUNet
from places2_train import unnormalize, MEAN, STDDEV

from sr_mask_generator import SRMaskGenerator

def exceeds_bounds(y):
    if y >= 250:
        return True
    else:
        return False

class Drawer(QWidget):
    newPoint = pyqtSignal(QPoint)
    def __init__(self, image_path, image_size, parent=None):
        QWidget.__init__(self, parent)
        self.path = QPainterPath()
        self.image_path = image_path
        self.image_size = image_size

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(QRect(0, 0, self.image_size, self.image_size), QPixmap(self.image_path))

        if (use_sr):
            return
        painter.setPen(QPen(Qt.black, 12))
        painter.drawPath(self.path)

    def mousePressEvent(self, event):
        if (exceeds_bounds(event.pos().y())
            or use_sr):
            return
        
        self.path.moveTo(event.pos())
        self.update()

    def mouseMoveEvent(self, event):
        if (exceeds_bounds(event.pos().y())
            or use_sr):
            return
        
        self.path.lineTo(event.pos())
        self.newPoint.emit(event.pos())
        self.update()

    def sizeHint(self):
        return QSize(self.image_size, self.image_size)

    def resetPath(self):
        self.path = QPainterPath()
        self.update()

class InpaintApp(QWidget):

    def __init__(self, image_num):
        super().__init__()
        self.setLayout(QVBoxLayout())

        self.title = 'Inpaint Application'
        self.width = 276
        self.height = 350
        self.cwd = os.getcwd()

        image_num = str(image_num).zfill(8)
        self.image_path = self.cwd + "/val_256/Places365_val_{}.jpg".format(image_num)
        img = Image.open(self.image_path)
        self.img_size = img.height

        if (use_sr):
            ori_shape = (self.img_size,self.img_size)
            downsampling_shape = ((int)(self.img_size/sr_rate), (int)(self.img_size/sr_rate))
            down_img = Image.new('RGB', downsampling_shape)
            for h in range (down_img.height):
                for w in range (down_img.width):
                    desired_pixel = img.getpixel((w * sr_rate, h * sr_rate))
                    down_img.putpixel((w, h), desired_pixel)
                   
            down_img = down_img.resize(ori_shape)
            self.image_path = self.cwd + "/downsample.png"
            down_img.save(self.image_path)
                    
        self.save_path = self.cwd + "/test.png"
        self.open_and_save_img(self.image_path, self.save_path)
        self.drawer = Drawer(self.save_path, self.img_size, self)

        self.setWindowTitle(self.title)
        self.setGeometry(200, 200, self.width, self.height)

        self.layout().addWidget(self.drawer)
        self.layout().addWidget(QPushButton("Inpaint!", clicked=self.inpaint))
        
        self.img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STDDEV)])
        self.mask_transform = transforms.ToTensor()
        self.device = torch.device("cpu")

        sr_mask_shape = (self.img_size,self.img_size)
        self.sr_mask_gen = SRMaskGenerator(sr_mask_shape, self.device, sr_rate, torch.float)

        if (use_sr):
            model_dict = torch.load(self.cwd + "/model/sr_model_e0_i500.pth", map_location="cpu")
        else:
            model_dict = torch.load(self.cwd + "/model/irr_model_e0_i500.pth", map_location="cpu")

        model = PartialConvUNet()
        model.load_state_dict(model_dict["model"])
        model = model.to(self.device)

        self.model = model
        self.model.eval()
        
        self.show()

    def open_and_save_img(self, path, dest):
        img = Image.open(path)
        img.save(dest)

    def inpaint(self):
        if (use_sr):
            mask = self.sr_mask_gen.get_sr_mask()
        else:
            mask = QImage(self.img_size, self.img_size, QImage.Format_RGB32)
            mask.fill(qRgb(255, 255, 255))

            painter = QPainter()
            painter.begin(mask)
            painter.setPen(QPen(Qt.black, 12))
            painter.drawPath(self.drawer.path)
            painter.end()

            mask.save("mask.png", "png")

            # open image and normalize before forward pass
            mask = Image.open(self.cwd + "/mask.png")
            mask = self.mask_transform(mask.convert("RGB"))

        gt_img = Image.open(self.save_path)
        gt_img = self.img_transform(gt_img.convert("RGB"))
        img = gt_img * mask

        # adds dimension of 1 (batch) to image
        img.unsqueeze_(0)
        gt_img.unsqueeze_(0)
        mask.unsqueeze_(0)

        # forward pass
        with torch.no_grad():
            output = self.model(img.to(self.device), mask.to(self.device))

        # unnormalize the image and output
        output = mask * img + (1 - mask) * output
        grid = make_grid(unnormalize(output))
        save_image(grid, "test.png")

        self.drawer.resetPath()
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=int, default=1)
    parser.add_argument('--super_resolution', '--sr', dest='use_sr', action='store_true')
    parser.add_argument('--sr_rate', type=int, default=2)
    args = parser.parse_args()

    use_sr = args.use_sr
    sr_rate = args.sr_rate
    app = QApplication(sys.argv)
    ex = InpaintApp(args.img)
    sys.exit(app.exec_())
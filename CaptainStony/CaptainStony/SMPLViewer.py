import sys
sys.path.append('../data/')
sys.path.append('../data/smpl/smpl_webuser/')
sys.path.append('../CaptainStony/')
sys.path.append('../CaptainStony/CaptainStony/ui')


import os
import pickle as pkl
import numpy as np
import render_model
from smpl.smpl_webuser.serialization import load_model
import cv2


import sys
from PyQt5 import QtWidgets
from ui.main_window import Ui_MainWindow

if __name__ == '__main__':

    def linspace(a, b, n=100):
        if n < 2:
            return b
        diff = int((float(b) - a)/(n - 1))
        return [diff * i + a  for i in range(int(n))]

    xs = linspace(1, int(600 / 32), int(600 / 32))
    ys = linspace(1, int(600 / 41), int(600 / 41))
    zs = linspace(1, int(600 / 73), int(600 / 73))

    rhs = linspace(15, 600, (600 - 15) / 15 + 1)

    for rh in rhs:
        for x in xs:
            for y in ys:
                for z in zs:
                    if 32 * x + 41 * y + 73 * z == rhs:
                        print("solution to {} is x = {}, y = {}, z = {}".format(rh, x, y, z))


    import logging

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')

    logging.info("working dir: {}".format(os.getcwd()))

    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = "H:/dev/ROMP/demo/images_results"
        
    logging.info("data dir: {}".format(data_path))
    os.environ["SMPL_VIEWER_DATA_DIR"] = data_path

    app = QtWidgets.QApplication(sys.argv)
    form = Ui_MainWindow()
    form.show()
    form.raise_()
    sys.exit(app.exec_())



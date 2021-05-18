import sys
sys.path.append('./data/')
sys.path.append('./data/smpl/smpl_webuser/')
sys.path.append('./CaptainStony/opendr/')
sys.path.append('./CaptainStony/CaptainStony/ui')


import os
import pickle as pkl
import numpy as np
import render_model
from smpl.smpl_webuser.serialization import load_model
import cv2


def renderImage(model,img_path,camPose,camIntrinsics):

    img = cv2.imread(img_path)
    class cam:
        pass
    cam.rt = cv2.Rodrigues(camPose[0:3,0:3])[0].ravel()
    cam.t = camPose[0:3,3]
    cam.f = np.array([camIntrinsics[0,0],camIntrinsics[1,1]])
    cam.c = camIntrinsics[0:2,2]
    h = int(2*cam.c[1])
    w = int(2*cam.c[0])
    im = (render_model.render_model(model, model.f, w, h, cam, img= img)* 255.).astype('uint8')
    return im


import sys
from PyQt5 import QtWidgets
from ui.main_window import Ui_MainWindow

if __name__ == '__main__':

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



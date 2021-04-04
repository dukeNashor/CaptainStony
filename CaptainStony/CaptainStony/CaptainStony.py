import sys
sys.path.append('./data/')
sys.path.append('./data/smpl/smpl_webuser/')
sys.path.append('./3rdparty/opendr/')
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

    print(os.getcwd())

    seq_name = 'courtyard_basketball_00'
    datasetDir = './data/3DPW'
    file = os.path.join(datasetDir, 'sequenceFiles', "train", seq_name+'.pkl')
    
    with open(file,'rb') as f:
        seq = pkl.load(f, encoding='latin1')

    models = list()
    for iModel in range(0,len(seq['v_template_clothed'])):
        if seq['genders'][iModel] == 'm':
            model = load_model("./data/smpl/models/basicmodel_m_lbs_10_207_0_v1.1.0.pkl")
        else:
            model = load_model("./data/smpl/models/basicModel_f_lbs_10_207_0_v1.1.0.pkl")

        model.betas[:10] = seq['betas'][iModel][:10]
        models.append(model)

    iModel = 0
    iFrame = 25
    if seq['campose_valid'][iModel][iFrame]:
        models[iModel].pose[:] = seq['poses'][iModel][iFrame]
        models[iModel].trans[:] = seq['trans'][iModel][iFrame]
        img_path = os.path.join(datasetDir,'imageFiles',seq['sequence']+'/image_{:05d}.jpg'.format(iFrame))
        im = renderImage(models[iModel],img_path,seq['cam_poses'][iFrame],seq['cam_intrinsics'])
        cv2.imshow('3DPW Example',im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    app = QtWidgets.QApplication(sys.argv)
    form = Ui_MainWindow()
    form.show()
    form.raise_()
    sys.exit(app.exec_())
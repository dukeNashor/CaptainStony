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
from copy import deepcopy


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


def Process3DPW(dir_3dpw, dir_smpl = None, out_dir = None, generate_file_list = True):

    if out_dir is None:
        out_dir = "./data/3dpw_processed/"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    if generate_file_list:
        file_list_image = open(os.path.join(out_dir, "image_file_list.txt"), 'w')
        file_list_verts = open(os.path.join(out_dir, "verts_file_list.txt"), 'w')


    image_path = os.path.join(dir_3dpw, "imageFiles")

    seq_path = os.path.join(dir_3dpw, "sequenceFiles")

    if not os.path.exists(image_path):
        raise ValueError("given path is not 3DPW dataset directory!")


    if dir_smpl is None:
        model_m = load_model("./data/smpl/models/basicmodel_m_lbs_10_207_0_v1.1.0.pkl")
        model_f = load_model("./data/smpl/models/basicModel_f_lbs_10_207_0_v1.1.0.pkl")
    else:
        model_m = load_model(os.path.join("./models/basicmodel_m_lbs_10_207_0_v1.1.0.pkl"))
        model_f = load_model(os.path.join("./models/basicModel_f_lbs_10_207_0_v1.1.0.pkl"))

    # all seq names
    seq_names = [f for f in os.listdir(image_path)]
    
    seq_file_paths = []
    for path, subdirs, files in os.walk(seq_path):
        for name in files:
            # check extension
            if os.path.splitext(name)[-1] == ".pkl":
                seq_file_paths.append(os.path.join(path, name))

    # process each seq
    for seq_name in seq_names:
        name = [ n for n in seq_file_paths if seq_name in n]

        if len(name) != 1:
            print("skipping {}: seq file not found".format(name))
            continue

        name = name[0]
        with open(name, 'rb') as f:
            seq = pkl.load(f, encoding='latin1')

        if len(seq['v_template_clothed']) > 1:
            print("skipping {}: {} ppl in seq.".format(name, str(len(seq['v_template_clothed']))))
            continue

        print("Processing {}".format(name))

        models = list()
        for iModel in range(0,len(seq['v_template_clothed'])):
            if seq['genders'][iModel] == 'm':
                model = deepcopy(model_m)
            else:
                model = deepcopy(model_f)

            model.betas[:10] = seq['betas'][iModel][:10]
            models.append(model)

        iModel = 0
        num_frames = seq['cam_poses'].shape[0]
        for iFrame in range(num_frames):
            if seq['campose_valid'][iModel][iFrame]:
                out_file_name = os.path.join(out_dir, seq['sequence']+'/image_{:05d}.npy'.format(iFrame))

                if generate_file_list:
                    img_path = os.path.join(dir_3dpw + '/imageFiles/',seq['sequence']+'/image_{:05d}.jpg'.format(iFrame))
                    file_list_image.write(img_path + "\n")
                    file_list_verts.write(out_file_name + "\n")
                    continue

                models[iModel].pose[:] = seq['poses'][iModel][iFrame]
                models[iModel].trans[:] = seq['trans'][iModel][iFrame]

                ## debug
                #img_path = os.path.join(dir_3dpw,'imageFiles',seq['sequence']+'/image_{:05d}.jpg'.format(iFrame))
                #im = renderImage(models[iModel],img_path,seq['cam_poses'][iFrame],seq['cam_intrinsics'])
                #cv2.imshow('3DPW Example',im)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()

                # save results
                if not os.path.exists(os.path.dirname(out_file_name)):
                    os.makedirs(os.path.dirname(out_file_name))
                
                out_file_name = os.path.join(out_dir, seq['sequence']+'/image_{:05d}.csv'.format(iFrame))
                np.savetxt(out_file_name, model.r)
                #np.save(out_file_name, models[iModel].r)


    if generate_file_list:
        file_list_image.close()
        file_list_verts.close()



if __name__ == '__main__':
    print(os.getcwd())

    dir_3dpw = './data/3DPW'
    Process3DPW(dir_3dpw, generate_file_list = False)

    
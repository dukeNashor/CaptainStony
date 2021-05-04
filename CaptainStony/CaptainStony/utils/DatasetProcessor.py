import os

import cv2
import numpy as np
import pickle

from CaptainStony.CaptainStony.MetroParams import *

def Preprocess3DPW(dataset_dir, out_dir, out_name = "3dpw_processed"):

    # scale factor
    factor_scale = 1.2

    image_names_, scales_, centers_, poses_, shapes_, genders_ = [], [], [], [], [], []

    # get pkl files in the sequenceFiles folder
    dataset_dir = os.path.join(dataset_dir, 'sequenceFiles', 'test')
    pickle_files = [os.path.join(dataset_dir, f) 
        for f in os.listdir(dataset_dir) if f.endswith('.pkl')]

    for filename in pickle_files:
        with open(filename, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            smpl_pose = data['poses']
            smpl_betas = data['betas']
            poses2d = data['poses2d']
            cam_poses = data['cam_poses']
            genders = data['genders']
            valid = np.array(data['campose_valid']).astype(np.bool_)
            num_people = len(smpl_pose)
            num_frames = len(smpl_pose[0])
            seq_name = str(data['sequence'])
            img_names = np.array(['imageFiles/' + seq_name + '/image_%s.jpg' % str(i).zfill(5) for i in range(num_frames)])
            # get through all the people in the sequence
            for i in range(num_people):
                valid_pose = smpl_pose[i][valid[i]]
                valid_betas = np.tile(smpl_betas[i][:10].reshape(1,-1), (num_frames, 1))
                valid_betas = valid_betas[valid[i]]
                valid_keypoints_2d = poses2d[i][valid[i]]
                valid_img_names = img_names[valid[i]]
                valid_global_poses = cam_poses[valid[i]]
                gender = genders[i]
                # consider only valid frames
                for valid_i in range(valid_pose.shape[0]):
                    part = valid_keypoints_2d[valid_i,:,:].T
                    part = part[part[:,2]>0,:]
                    bbox = [min(part[:,0]), min(part[:,1]),
                        max(part[:,0]), max(part[:,1])]
                    center = [(bbox[2] + bbox[0])/2, (bbox[3] + bbox[1])/2]
                    scale = factor_scale*max(bbox[2] - bbox[0], bbox[3] - bbox[1])/200
                    
                    # transform global pose
                    pose = valid_pose[valid_i]
                    extrinsics = valid_global_poses[valid_i][:3,:3]
                    pose[:3] = cv2.Rodrigues(np.dot(extrinsics, cv2.Rodrigues(pose[:3])[0]))[0].T[0]                      

                    image_names_.append(valid_img_names[valid_i])
                    centers_.append(center)
                    scales_.append(scale)
                    poses_.append(pose)
                    shapes_.append(valid_betas[valid_i])
                    genders_.append(gender)

    # serialization
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    out_file = os.path.join(out_dir, out_name)

    dict_save = { "image_names": image_names_,
                  "centers": centers_,
                  "scales": scales_,
                  "poses": poses_,
                  "shapes": shapes_,
                  "genders": genders_
                  }

    np.savez(out_file, **dict_save)
                       
                       
                       
                       
                       



# test
if __name__ == "__main__":


    Preprocess3DPW(dataset_dir = PATH_3DPW, out_dir = PATH_PROCESSED_3DPW)

    processed_3dpw_path = os.path.join(PATH_PROCESSED_3DPW, "3dpw_processed.npz")

    data = np.load(processed_3dpw_path)



    pass
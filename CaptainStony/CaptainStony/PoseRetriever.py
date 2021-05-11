import numpy as np
from PIL import Image

import pickle as pkl
import logging
import os
import copy

from scipy.spatial.transform import Rotation as R
import scipy.spatial.distance as ssd

MASK_USE_ALL = 1
MASK_IGNORE_FAR_END = 2
MASK_USE_TRUNK_ONLY = 3

mask_use_all = np.ones(72, dtype = np.float32)
mask_use_all[0:3] = 0.0
mask_ignore_far_end = copy.deepcopy(mask_use_all)
mask_ignore_far_end[60:72] = 0.0

mask_use_trunk_only = copy.deepcopy(mask_use_all)
mask_use_trunk_only[21:36] = 0.0
mask_use_trunk_only[60:72] = 0.0

mask_dict = {
    MASK_USE_ALL : mask_use_all,
    MASK_IGNORE_FAR_END : mask_ignore_far_end,
    MASK_USE_TRUNK_ONLY : mask_use_trunk_only
   }

DISTANCE_L1_NORM = 1
DISTANCE_L2_NORM = 2
DISTANCE_CANBERRA = 3
DISTANCE_CHEBYSHEV = 4
DISTANCE_COSINE = 5


class PoseRetrievalAlgorithm():

    @staticmethod
    def GetMask(type = MASK_USE_ALL):
        return mask_dict[type]

    @staticmethod
    def DistanceL1Norm(pose_1, pose_2, mask):
        return np.sum((np.multiply(pose_1, mask) - np.multiply(pose_2, mask)))

    @staticmethod
    def DistanceL2Norm(pose_1, pose_2, mask):
        return np.sum((np.multiply(pose_1, mask) - np.multiply(pose_2, mask)) ** 2)

    @staticmethod
    def DistanceCanberra(pose_1, pose_2, mask):
        return ssd.canberra(np.multiply(pose_1, mask), np.multiply(pose_2, mask))

    @staticmethod
    def DistanceChebyshev(pose_1, pose_2, mask):
        return ssd.chebyshev(np.multiply(pose_1, mask), np.multiply(pose_2, mask))

    @staticmethod
    def DistanceCosine(pose_1, pose_2, mask):
        return ssd.cosine(np.multiply(pose_1, mask), np.multiply(pose_2, mask))


class PoseRetriever(object):

    def __init__(self, load_image_into_memory = False):
        super(self.__class__, self).__init__()
        logging.info("PoseRetriever:initializing with load_image_into_memory = {}".format(load_image_into_memory))

        self.load_image_into_memory = load_image_into_memory
        self.image_dict = {}
        self.meta_dict = {}
        self.processed_dir = None
        self.distance_type = DISTANCE_L2_NORM
        self.distance_func = PoseRetrievalAlgorithm.DistanceL2Norm
        self.mask_type = MASK_USE_ALL
        self.mask = mask_dict[self.mask_type]

    def Load3DPWProcessedData(self, processed_dir):
        logging.info("PoseRetriever:Load3DPWProcessedData() started")
        self.processed_dir = processed_dir
        files = [f for f in os.listdir(processed_dir) if os.path.isfile(os.path.join(processed_dir, f))]

        for fname in files:
            [seq_name, img_name] = fname.split("-")
            processed_img_name = os.path.join(processed_dir, "data-3DPW/imageFiles", seq_name, img_name).replace('.pkl','.jpg')
            fname_clean = fname.replace('.pkl','')
            if not os.path.exists(processed_img_name):
                logging.info("file missing for " + processed_img_name)
                continue

            with open(os.path.join(processed_dir, fname), 'rb') as f:
                self.meta_dict[fname_clean] = pkl.load(f)
                for i, data in self.meta_dict[fname_clean].items():
                    data["pose"] = data["pose"].astype(np.float32)
            
            if self.load_image_into_memory:
                self.image_dict[fname_clean] = self.LoadImageFromFile(processed_img_name)
            else:
                self.image_dict[fname_clean] = processed_img_name

        logging.info("PoseRetriever:Load3DPWProcessedData() completed, loaded {} records.".format(len(self.image_dict)))

        
    def SetDistanceType(self, distance_type):
        self.distance_type = distance_type

        if distance_type == DISTANCE_L1_NORM:
            self.distance_func = PoseRetrievalAlgorithm.DistanceL1Norm
            return
        elif distance_type == DISTANCE_L2_NORM:
            self.distance_func = PoseRetrievalAlgorithm.DistanceL2Norm
            return
        elif distance_type == DISTANCE_CANBERRA:
            self.distance_func = PoseRetrievalAlgorithm.DistanceCanberra
            return
        elif distance_type == DISTANCE_CHEBYSHEV:
            self.distance_func = PoseRetrievalAlgorithm.DistanceChebyshev
            return
        elif distance_type == DISTANCE_COSINE:
            self.distance_func = PoseRetrievalAlgorithm.DistanceCosine
            return

    def SetMaskType(self, mask_type):
        self.mask_type = mask_type
        self.mask = mask_dict[self.mask_type]


    def GetDataList(self):
        datalist = []
        for name, data_dict in self.meta_dict.items():
            datalist.append(name)
        return datalist


    def Query(self, pose):
        logging.info("PoseRetriever:Query()")

        results = []

        for name, data_dict in self.meta_dict.items():
            best_result_in_image = 10e4
            for id, meta_data in data_dict.items():
                score = self.distance_func(pose, meta_data["pose"], self.mask)
                if score < best_result_in_image:
                    best_result_in_image = score

            results.append((name, score))
        
        results.sort(key = lambda x: x[1])

        logging.info("PoseRetriever:Query() completed")
        return results

    
    def GetPose(self, key):
        pose_rv = self.meta_dict[key][0]["pose"]
        pose_ea = pose_rv.copy()
        for i in range(24):
            rot = R.from_rotvec(pose_rv[i*3 : i*3+3])
            ea = rot.as_euler("xyz", degrees = False)
            pose_ea[i*3], pose_ea[i*3 + 1], pose_ea[i*3 + 2] = ea[0], ea[1], ea[2]

        ##### Debug code to check consistency of scipy rotation conversions
        #pose_recover = pose_rv.copy()
        #for i in range(24):
        #    rot = R.from_euler(seq = "xyz", angles = pose_ea[i*3 : i*3+3])
        #    rv = rot.as_rotvec()
        #    pose_recover[i*3], pose_recover[i*3 + 1], pose_recover[i*3 + 2] = rv[0], rv[1], rv[2]

        #print("original:")
        #print(pose_rv)
        #print("recovered:")
        #print(pose_recover)
        #print("L: ")
        #print(pose_ea[54:57])
        #print("R: ")
        #print(pose_ea[57:60])

        return pose_ea

    def GetImage(self, key):
        if key not in self.image_dict:
            return None
        elif self.load_image_into_memory:
            return self.image_dict[key]
        else:
            return self.LoadImageFromFile(self.GetActualPath(key))


    @staticmethod
    def LoadImageFromFile(image_file_name):
        return np.asarray(Image.open(image_file_name))


    def GetActualPath(self, key):
        [seq_name, img_name] = key.split("-")
        actual_path = os.path.join(self.processed_dir, "data-3DPW/imageFiles", seq_name, img_name) + '.jpg'
        return actual_path


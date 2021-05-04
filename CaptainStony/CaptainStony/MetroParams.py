import numpy as np


# regressor matrix downloaded from: https://github.com/nkolot/SPIN/blob/master/fetch_data.sh
G_JOINT_REGRESSOR_PATH = "./data/J_regressor_extra.npy"

regressor_matrix = np.load(G_JOINT_REGRESSOR_PATH)

NUM_JOINTS = 24
NUM_VERTICES = 400




# path of datasets
PATH_3DPW = "./data/3DPW/"
PATH_PROCESSED_3DPW = "./data/3DPW_processed/"
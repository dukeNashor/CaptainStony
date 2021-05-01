import numpy as np


# regressor matrix downloaded from: https://github.com/nkolot/SPIN/blob/master/fetch_data.sh
G_JOINT_REGRESSOR_PATH = "./data/J_regressor_extra.npy"

regressor_matrix = np.load(G_JOINT_REGRESSOR_PATH)

NUM_JOINTS = 21
NUM_VERTICES = 400
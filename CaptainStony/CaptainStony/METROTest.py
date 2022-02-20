import torch
import torch.nn as nn
import numpy as np

import sys
import os
sys.path.append('./CaptainStony/CaptainStony/')
os.environ['TORCH_HOME'] = 'Cache'

from model.ImageFeatureExtraction import CNN
from model.MeshPositionalEncoding import MeshPositionalEncoding
from model.MultiLayerTransformer import MultiLayerEncoder, FuncLoss
from model.Metro import Metro

def GetFileList(image_file_list_name, verts_file_list_name):

    with open(image_file_list_name) as f:
        image_file_list = f.read().splitlines()

    with open(verts_file_list_name) as f:
        verts_file_list = f.read().splitlines()

    return image_file_list, verts_file_list


class VertexSelector():

    def __init__(self, mesh_path = "./data/SMPL_template_simplified_vertices.csv"):
        # load a few vertices
        csv_data = np.genfromtxt(mesh_path, delimiter=' ')
        self.verts = csv_data[:, 0:3]
        self.indices = csv_data[:,-1].astype(np.int32)
        self.num_vertices = self.verts.shape[0]

    def GetVertices(self, verts_ori):
        return verts_ori[self.indices, :]


BATCH_SIZE = 1
NUM_EPOCHS = 5
LEARNING_RATE = 0.001
SAVE_VERTS_N_FRAME = 100
SAVE_MODEL_N_FRAME = 10000

if __name__ == "__main__":
    from torchvision import datasets, transforms
    import matplotlib.pyplot as plt
    from PIL import Image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Metro(batch_size = BATCH_SIZE, device = device)
    
    verts_selector = VertexSelector()

    dir_3dpw = './data/3DPW'

    dir_debug = "./debug/"
    if not os.path.exists(dir_debug):
        os.makedirs(dir_debug)

    dir_checkpoint = "./checkpoints/"
    if not os.path.exists(dir_checkpoint):
        os.makedirs(dir_checkpoint)

    #model.load_state_dict(torch.load(os.path.join(dir_checkpoint, "020000.pth")))
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    image_file_list, verts_file_list = GetFileList("./data/3dpw_processed/image_file_list.txt",
                                                   "./data/3dpw_processed/verts_file_list.txt")

    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.ToTensor()])


    import torch.optim as optim

    funcloss = FuncLoss(device)
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

    total_n = 0
    for epoch in range(NUM_EPOCHS):
        for fname_im, fname_vt in zip(image_file_list, verts_file_list):
            print("Processing {}".format(os.path.basename(fname_im)))

            image = Image.open(os.path.join(fname_im))
            verts = verts_selector.GetVertices(np.load(fname_vt))
            mesh_verts = torch.from_numpy(verts).to(device)
            
            image_transformed = transform(image).to(device)
            if image_transformed.ndim < 4:
                image_transformed = image_transformed.unsqueeze(0)

            # forward
            out_feature = model.forward(image_transformed)
            loss = funcloss(out_feature, mesh_verts)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # step
            optimizer.step()

            # save image
            if total_n % SAVE_VERTS_N_FRAME == 0:
                np.savetxt(dir_debug + os.path.basename(fname_im) + "_{:06d}.csv".format(total_n), out_feature.detach().cpu().numpy().squeeze(0))

            if total_n % SAVE_MODEL_N_FRAME == 0:
                torch.save(model, os.path.join(dir_checkpoint, "{:06d}.pth".format(total_n)))

            total_n = total_n + 1

            

    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
import trimesh
import numpy as np
import torch
import torch.nn as nn

import os
import logging

class MeshPositionalEncoding(nn.Module):
    def __init__(self, batch_size, device = 'cpu', mesh_path = "./data/SMPL_template_mesh.obj"):
        super(MeshPositionalEncoding, self).__init__()

        if (os.path.splitext(mesh_path)[-1] == ".obj"):
            try:
                self.template_mesh = trimesh.load(mesh_path)
                self.positional_encoding_mat = torch.from_numpy(np.array(self.template_mesh.vertices).astype(np.float32)).repeat(batch_size, 1, 1).to(device)
                self.num_vertices = len(self.template_mesh.vertices)
            except:
                logging.info("Mesh", mesh_path, "does not exist")
                self.template_mesh = None
        elif (os.path.splitext(mesh_path)[-1] == ".csv"):
            # load a few vertices
            csv_data = np.genfromtxt(mesh_path, delimiter=' ')
            verts = csv_data[:, 0:3]
            self.positional_encoding_mat = torch.from_numpy(verts.astype(np.float32)).repeat(batch_size, 1, 1).to(device)
            self.num_vertices = verts.shape[0]


        self.batch_size = batch_size
        

    def forward(self, input):
        reped = input.repeat(1, self.num_vertices, 1)
        out = torch.cat(tensors = (self.positional_encoding_mat, reped), dim = 2)
        return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mpe = MeshPositionalEncoding(mesh_path = "./data/SMPL_template_simplified_vertices.csv")
    print(mpe.positional_encoding_mat.shape)

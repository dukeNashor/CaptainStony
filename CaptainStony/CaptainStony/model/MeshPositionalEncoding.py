import trimesh
import numpy as np
import torch
import torch.nn as nn

import logging

class MeshPositionalEncoding(nn.Module):
    def __init__(self, mesh_path = "./data/SMPL_template_mesh.obj"):
        super(MeshPositionalEncoding, self).__init__()
        try:
            self.template_mesh = trimesh.load(mesh_path)
        except:
            logging.info("Mesh", mesh_path, "does not exist")
            self.template_mesh = None

        self.positional_encoding = np.array(self.template_mesh.vertices)

    def forward(self, input):
        out = self.positional_encoding + input
        return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mpe = MeshPositionalEncoding()
    print(mpe.positional_encoding.shape)

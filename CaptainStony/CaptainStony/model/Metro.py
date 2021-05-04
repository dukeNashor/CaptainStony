import torch
import torch.nn as nn

import sys
import os
sys.path.append('./CaptainStony/CaptainStony/')
os.environ['TORCH_HOME'] = 'Cache'

from model.ImageFeatureExtraction import CNN
from model.MeshPositionalEncoding import MeshPositionalEncoding
from model.MultiLayerTransformer import MultiLayerEncoder

class Metro(nn.Module):
    def __init__(self, batch_size, device = 'cpu'):
        super(Metro, self).__init__()

        self.device = device

        self.cnn = CNN(batch_size = batch_size, device = device)
        self.pe = MeshPositionalEncoding(batch_size = batch_size, device = device, mesh_path = "./data/SMPL_template_mesh_simplified.obj")

        reduced_dims = [2048 // 2, 2048 // 4, 2048 // 8, 3]

        embed_dim = self.pe.num_vertices + 3

        self.mle = MultiLayerEncoder(embed_size = 2051, reduced_dims = reduced_dims, device = device)


    def forward(self, inputs):
        out = self.cnn(inputs)
        out = self.pe(out)
        out = self.mle(out)
        return out


if __name__ == "__main__":

    BATCH_SIZE = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metro = Metro(batch_size = BATCH_SIZE, device = device)

    from torchvision import datasets, transforms
    import matplotlib.pyplot as plt

    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.ToTensor()])

    dataset = datasets.ImageFolder('./data/3DPW/imageFiles/', transform = transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = False)
    images, labels = next(iter(dataloader))
    print("showing first image of size:", *images[0].shape)
    plt.imshow(images[0].permute(1, 2, 0))
    plt.show()

    out_feature = metro.forward(images.to(device))


    pass
import torch
import torch.nn as nn

from model.ImageFeatureExtraction import CNN
from model.MeshPositionalEncoding import MeshPositionalEncoding
from model.MultiLayerTransformer import MultiLayerEncoder

class Metro(nn.Module, device = 'cpu'):
    def __init__(self):
        super(Metro, self).__init__()
        self.cnn = CNN(device = device)
        self.mle = MultiLayerEncoder(device = device)


    def forward(self, inputs):
        
        out = self.cnn(inputs)
        out = self.mle(inputs)
        return out


if __name__ == "__main__":


    pass
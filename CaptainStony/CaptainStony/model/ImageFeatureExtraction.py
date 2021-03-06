import torch
import torch.nn as nn
from torchvision import models

import logging

class CNN(nn.Module):
    def __init__(self, batch_size, device = "cpu"):
        super(CNN, self).__init__()
        self.batch_size = batch_size
        self.device = device
        resnet_model = models.resnet50(pretrained = True)
        self.resnet = torch.nn.Sequential(*(list(resnet_model.children())[:-1]))
        self.resnet.train(False)
        self.to(self.device)
        logging.info(self.resnet)
        self.resnet.eval()

    def forward(self, inputs):
        out = self.resnet(inputs)
        out = out.reshape((self.batch_size, 1, -1))
        return out






# test code

if __name__ == "__main__":

    BATCH_SIZE = 16

    import os
    
    from torchvision import datasets, transforms
    import matplotlib.pyplot as plt

    logging.basicConfig(level=logging.INFO)

    os.environ['TORCH_HOME'] = 'Cache'
    print(os.getcwd())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cnn = CNN(device)


    #transform = transforms.Compose([transforms.Resize(255),
    #                                transforms.CenterCrop(224),
    #                                transforms.ToTensor()])

    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.ToTensor()])

    # ImageFolder only supports formats like : [dataset/dog] [dataset/cat]
    # dataset = datasets.ImageFolder('./data/3DPW/imageFiles/courtyard_arguing_00/', transform = transform)

    dataset = datasets.ImageFolder('./data/3DPW/imageFiles/', transform = transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = False)
    images, labels = next(iter(dataloader))
    print("showing first image of size:", *images[0].shape)
    plt.imshow(images[0].permute(1, 2, 0))
    plt.show()

    out_feature = cnn.forward(images)

    pass
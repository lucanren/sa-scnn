import torch
import torch.nn as nn
import numpy as np

class net_one_neuron(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=30, kernel_size=(5, 5), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(30),
            nn.Sigmoid(),
            nn.Dropout2d(0.3),
            nn.Conv2d(in_channels=30, out_channels=30, kernel_size=(5, 5), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(30),
            nn.Sigmoid(),
            nn.Dropout2d(0.3),
            nn.Conv2d(in_channels=30, out_channels=30, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(30),
            nn.Sigmoid(),
            nn.Dropout2d(0.3), #or here
            nn.Conv2d(in_channels=30, out_channels=30, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(30),
            nn.Sigmoid(),
        )
        # self.layers_more = nn.Sequential(
        #     nn.Conv2d(in_channels=30, out_channels=30, kernel_size=(3, 3), stride=(1, 1)),
        #     nn.BatchNorm2d(30),
        #     nn.Sigmoid(),
        #     nn.Conv2d(in_channels=30, out_channels=30, kernel_size=(3, 3), stride=(1, 1)),
        #     nn.BatchNorm2d(30),
        #     nn.Sigmoid()
        # )
        self.flatten = nn.Flatten()
        self.Linear = nn.Linear(5 * 5 * 30, 1)

    def forward(self, x):
        x = self.layers(x)
        # x = self.layers_more(x)
        x = self.flatten(x)
        x = self.Linear(x)
        return x


class seperate_core_model(nn.Module):
    def __init__(self,num_neurons):
        super().__init__()
        self.models = nn.ModuleList([net_one_neuron() for i in range(num_neurons)])
        self.num_neurons = num_neurons

    def forward(self, x):
        outputs = [self.models[i].forward(x) for i in range(self.num_neurons)]
        outputs = torch.stack(outputs, dim=1)
        return outputs.reshape((outputs.shape[0], outputs.shape[1]))
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import math
import cv2
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt


class BinaryClassifier(nn.Module):
    def __init__(self, input_dim = 4*33):
        super(BinaryClassifier, self).__init__()
        
        self.input_dim = input_dim
        
        self.fc1 = nn.Linear(input_dim, 64)
        self.dropout1 = nn.Dropout(p=0.5) # Add dropout layer with p=0.5
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(p=0.5) # Add another dropout layer with p=0.5
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.dropout1(x) # Add dropout layer after first fully connected layer
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.dropout2(x) # Add dropout layer after second fully connected layer
        x = self.fc3(x)
        x = nn.ReLU()(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x
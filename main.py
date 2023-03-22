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
from data_pre import data_loaders
from model import BinaryClassifier, test,train

device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
input_dim = 10
model = BinaryClassifier(input_dim).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
alpha = 0.01
train_dataloader, test_dataloader = data_loaders("train-path","test-path")

train_loss_history = []
test_loss_history = []
test_acc_history = []


for epoch in range(num_epochs):
    train_loss = train(model, train_dataloader, criterion, optimizer, device, alpha)
    train_loss_history.append(train_loss)

    test_loss, test_acc = test(model, test_dataloader, criterion, device)
    test_loss_history.append(test_loss)
    test_acc_history.append(test_acc)

    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')

def inference(model, inputs, threshold=0.5):
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        predicted = (outputs >= threshold).squeeze().long()
    return predicted




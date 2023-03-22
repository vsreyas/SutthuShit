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
    def __init__(self, input_dim):
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

def train(model, dataloader, criterion, optimizer, device, alpha=0.01):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Add L2 regularization to the loss
        l2_reg = torch.tensor(0.).to(device)
        for name, param in model.named_parameters():
            if 'weight' in name:
                l2_reg += torch.norm(param, p=2)**2
        loss += alpha * l2_reg

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)

def test(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            predicted = (outputs >= 0.5).squeeze().long()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return running_loss / len(dataloader), accuracy

device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
input_dim = 10
model = BinaryClassifier(input_dim).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
alpha = 0.01
# Define the transforms to be applied to the images
class MyTransform:
    def __init__(self, scale_factor):
        self.scale_factor = scale_factor

    def __call__(self, img):
        # Initializing mediapipe pose class.
        mp_pose = mp.solutions.pose

        # Setting up the Pose function.
        pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

        # Initializing mediapipe drawing class, useful for annotation.
        mp_drawing = mp.solutions.drawing_utils
        sample_img = img

        x = []
        # Perform pose detection after converting the image into RGB format.
        results = pose.process(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))
        image_height, image_width, _ = sample_img.shape
        # Check if any landmarks are found.
        if results.pose_landmarks:
            # Iterate two times as we only want to display first two landmark.
            for i in range(33):
                # Display the found landmarks after converting them into their original scale.
                x.extend([results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].x * image_width, results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].y *image_height, results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].z * image_width, results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].visibility])
        # x is a one-dimensional vector
        return x * self.scale_factor
    
transform = transforms.Compose([
    MyTransform(scale_factor=2),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
# Define the paths to the train and test data directories
train_data_path = 'path/to/train/data'
test_data_path = 'path/to/test/data'

# Create the train and test datasets using the ImageFolder class
train_dataset = datasets.ImageFolder(train_data_path, transform=transform)
test_dataset = datasets.ImageFolder(test_data_path, transform=transform)

# Define the batch size for the dataloaders
batch_size = 32

# Create the train and test dataloaders
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
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




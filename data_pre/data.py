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
    
def data_loaders(train_path, test_path, batch_size = 32,scale =2 ):    
    transform = transforms.Compose([
        MyTransform(scale_factor=scale),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    # Define the paths to the train and test data directories
    train_data_path = train_path
    test_data_path = test_path

    # Create the train and test datasets using the ImageFolder class
    train_dataset = datasets.ImageFolder(train_data_path, transform=transform)
    test_dataset = datasets.ImageFolder(test_data_path, transform=transform)

    # Create the train and test dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader,test_dataloader



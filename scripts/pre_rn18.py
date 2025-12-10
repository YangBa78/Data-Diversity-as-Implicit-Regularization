import torch
import torchvision
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import MulticlassCalibrationError
from tqdm import tqdm
from pyhessian import hessian
import numpy as np

# Part 1: Dataset Class
class CustomDataset(Dataset):
    def __init__(self, images, labels, classes, transform=None):
        self.images = images
        self.labels = labels
        self.classes = classes  # List of class names passed as input
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# Part 2: Fine-tuner Class with ResNet18
class ResNetFinetuner(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.0, resnet_model="resnet18", pretrained=True):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load a ResNet model
        self.resnet = getattr(torchvision.models, resnet_model)(pretrained=pretrained).to(self.device)
        
        # Extract the input features of the final fully connected layer
        in_features = self.resnet.fc.in_features
        
        # Replace the fully connected layer with an Identity layer
        self.resnet.fc = nn.Identity()  
        
        # Add dropout and a new classifier layer
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # Pass through ResNet backbone
        features = self.resnet(x)
        
        # Apply dropout and classifier
        features = self.dropout(features)
        return self.classifier(features)



# Part 3: Train Function (Updated with Training ECE)
def eigen_train(model, train_loader, num_epochs, lr=1e-5, weight_decay=0.0):
    model.to(model.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

  
    trace_train = []


    for _ in range(num_epochs):
        model.train()

        for inputs, labels in tqdm(train_loader):
            hessian_comp = hessian(model, criterion, data=(inputs, labels), cuda=True)
            trace_train.append(np.mean(hessian_comp.trace()).item())


            inputs, labels = inputs.to(model.device), labels.to(model.device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    return model, trace_train



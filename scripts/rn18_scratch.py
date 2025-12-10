import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
# from clip import clip
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import MulticlassCalibrationError
from tqdm import tqdm
from datasets import load_dataset
import torchvision.datasets as datasets
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from pyhessian import hessian
import numpy as np

from vendi_score import image_utils
import weightwatcher as ww


# Define the ResNet18 model as provided above
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.0):  # Added dropout_rate
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(p=dropout_rate)  # Added dropout layer
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.dropout(out)  # Apply dropout
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.0):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2, dropout_rate=dropout_rate)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2, dropout_rate=dropout_rate)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(p=dropout_rate)  # Dropout before the fully connected layer

    def _make_layer(self, block, out_channels, num_blocks, stride, dropout_rate):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, dropout_rate))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)  # Apply dropout
        out = self.fc(out)
        return out


class CustomDataset(Dataset):
    def __init__(self, images, labels, classes, transform=None):
        self.images = images
        self.labels = labels
        self.classes = classes
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    

def train(model, train_loader, val_loader, num_epochs, lr=1e-5, weight_decay=0.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay = weight_decay)

    num_classes = model.fc.out_features
    train_calibration_error = MulticlassCalibrationError(num_classes=num_classes, n_bins=15).to(device)

    trace_train = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        all_train_preds = []
        all_train_labels = []

        for inputs, labels in tqdm(train_loader):

            inputs, labels = inputs.to(device), labels.to(device)


            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            all_train_preds.append(outputs.softmax(dim=1))
            all_train_labels.append(labels)

        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total

        # Calculate training ECE
        all_train_preds = torch.cat(all_train_preds, dim=0)
        all_train_labels = torch.cat(all_train_labels, dim=0)
        train_ece = train_calibration_error(all_train_preds, all_train_labels)

        # Validation
        val_loss, val_acc, val_ece = evaluate(model, val_loader)

        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.3f}, Train Accuracy: {train_acc:.3f}%')
        print(f'Train ECE: {train_ece:.3f}')
        print(f'Val Loss: {val_loss:.3f}, Val Accuracy: {val_acc:.3f}%')
        print(f'Val ECE: {val_ece:.3f}')

    return model


# Part 4: Evaluate Function
def evaluate(model, data_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    total = 0

    num_classes = model.fc.out_features
    calibration_error = MulticlassCalibrationError(num_classes=num_classes, n_bins=15).to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.append(outputs.softmax(dim=1))
            all_labels.append(labels)

    avg_loss = total_loss / len(data_loader)
    accuracy = 100. * correct / total

    # Calculate calibration error
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    ece = calibration_error(all_preds, all_labels)

    return avg_loss, accuracy, ece.item()
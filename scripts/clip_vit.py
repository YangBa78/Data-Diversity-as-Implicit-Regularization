import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from scripts.clip_vit import clip
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import MulticlassCalibrationError
from tqdm import tqdm
from datasets import load_dataset
import torchvision.datasets as datasets

from vendi_score import image_utils
import weightwatcher as ww

# Part 1: Dataset Class
class CustomDataset(Dataset):
    def __init__(self, images, labels, classes, transform=None):
        """
        Args:
            images (list or array-like): A list or array of images.
            labels (list or array-like): A list or array of corresponding labels.
            classes (list): A list of class names.
            transform (callable, optional): A function/transform that takes in an image and returns a transformed version.
        """
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


# Part 2: ClipFinetuner Class
class ClipFinetuner(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.0, dp_attention = False, dp_mlp = False, clip_model="ViT-B/32"):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, _ = clip.load(clip_model, device=self.device)
        self.clip_model = self.clip_model.float()

        self.apply_dropout_to_attention = dp_attention
        self.apply_dropout_to_mlp = dp_mlp

        # Unfreeze all parameters in the visual encoder
        for param in self.clip_model.visual.parameters():
            param.requires_grad = True

        # Keep the text encoder frozen
        for param in self.clip_model.transformer.parameters():
            param.requires_grad = False

        # Add dropout to the encoder blocks in the visual part
        self.add_dropout_to_visual_encoder(dropout_rate)

        # New classifier layer
        self.classifier_dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.clip_model.visual.output_dim, num_classes)

    def add_dropout_to_visual_encoder(self, dropout_rate):
        vit = self.clip_model.visual
        for block in vit.transformer.resblocks:

            # Add dropout after the multi-head attention
            if self.apply_dropout_to_attention:
                block.attn_dropout = nn.Dropout(dropout_rate)

            # Add dropout after the MLP if specified
            if self.apply_dropout_to_mlp:
                block.mlp_dropout = nn.Dropout(dropout_rate)


    def forward(self, image):
        features = self.clip_model.encode_image(image).float()
        dropout = self.classifier_dropout(features)
        return self.classifier(dropout)

# Part 3: Train Function (Updated with Training ECE)
def train(model, train_loader, val_loader, num_epochs, lr=1e-5, weight_decay=0.0):
    model.to(model.device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay = weight_decay)

    num_classes = model.classifier.out_features
    train_calibration_error = MulticlassCalibrationError(num_classes=num_classes, n_bins=15).to(model.device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        all_train_preds = []
        all_train_labels = []

        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(model.device), labels.to(model.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
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
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%')
        print(f'Train ECE: {train_ece:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%')
        print(f'Val ECE: {val_ece:.4f}')

    return model

# Part 4: Evaluate Function
def evaluate(model, data_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    total = 0

    num_classes = model.classifier.out_features
    calibration_error = MulticlassCalibrationError(num_classes=num_classes, n_bins=15).to(model.device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(model.device), labels.to(model.device)
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
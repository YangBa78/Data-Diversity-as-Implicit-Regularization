import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from torch import nn
# import torch.optim as optim
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchmetrics.classification import MulticlassCalibrationError


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):

        self.labels = df['label'].tolist()
        self.texts = [tokenizer(text,
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in df['text']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):

        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):

        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y

class BertClassifier(nn.Module):
    def __init__(self, dropout=0.0, dp_attention=False, dp_mlp=False):
        super(BertClassifier, self).__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.apply_dropout_to_attention = dp_attention
        self.apply_dropout_to_mlp = dp_mlp
        
        # Apply dropout to attention and MLP layers if specified
        self.add_dropout_to_bert(dropout)
        
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 10)
        self.relu = nn.ReLU()
    
    def add_dropout_to_bert(self, dropout_rate):
        for layer in self.bert.encoder.layer:
            if self.apply_dropout_to_attention:
                layer.attention.self.dropout = nn.Dropout(dropout_rate)
            if self.apply_dropout_to_mlp:
                layer.output.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        
        return final_layer


# def train(model, train_loader, val_loader, num_epochs, lr=1e-5, weight_decay=0.0):
#     model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
#     num_classes = model.linear.out_features
#     train_calibration_error = MulticlassCalibrationError(num_classes=num_classes, n_bins=15).to(model.device)
    
#     for epoch in range(num_epochs):
#         model.train()
#         train_loss = 0.0
#         train_correct = 0
#         train_total = 0
        
#         all_train_preds = []
#         all_train_labels = []
        
#         for inputs, labels in tqdm(train_loader):
#             input_id, mask = inputs["input_ids"].squeeze(1).to(model.device), inputs["attention_mask"].squeeze(1).to(model.device)
#             labels = labels.to(model.device)
            
#             optimizer.zero_grad()
#             outputs = model(input_id, mask)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
            
#             train_loss += loss.item()
#             _, predicted = outputs.max(1)
#             train_total += labels.size(0)
#             train_correct += predicted.eq(labels).sum().item()
            
#             all_train_preds.append(outputs.softmax(dim=1))
#             all_train_labels.append(labels)
        
#         train_loss /= len(train_loader)
#         train_acc = 100. * train_correct / train_total
#         all_train_preds = torch.cat(all_train_preds, dim=0)
#         all_train_labels = torch.cat(all_train_labels, dim=0)
#         train_ece = train_calibration_error(all_train_preds, all_train_labels)
        
#         val_loss, val_acc, val_ece = evaluate(model, val_loader)
        
#         print(f'Epoch [{epoch+1}/{num_epochs}]')
#         print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%')
#         print(f'Train ECE: {train_ece:.4f}')
#         print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%')
#         print(f'Val ECE: {val_ece:.4f}')
    
#     return model

# def evaluate(model, data_loader):
#     model.eval()
#     criterion = nn.CrossEntropyLoss()
#     total_loss = 0
#     correct = 0
#     total = 0
    
#     num_classes = model.linear.out_features
#     calibration_error = MulticlassCalibrationError(num_classes=num_classes, n_bins=15).to(model.device)
    
#     all_preds = []
#     all_labels = []
    
#     with torch.no_grad():
#         for inputs, labels in data_loader:
#             input_id, mask = inputs["input_ids"].squeeze(1).to(model.device), inputs["attention_mask"].squeeze(1).to(model.device)
#             labels = labels.to(model.device)
            
#             outputs = model(input_id, mask)
#             loss = criterion(outputs, labels)
            
#             total_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += labels.size(0)
#             correct += predicted.eq(labels).sum().item()
            
#             all_preds.append(outputs.softmax(dim=1))
#             all_labels.append(labels)
    
#     avg_loss = total_loss / len(data_loader)
#     accuracy = 100. * correct / total
    
#     all_preds = torch.cat(all_preds, dim=0)
#     all_labels = torch.cat(all_labels, dim=0)
#     ece = calibration_error(all_preds, all_labels)
    
#     return avg_loss, accuracy, ece.item()


def train(model, train_data, val_data, learning_rate, epochs, weight_decay=0.0):

    train, val = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=32)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr= learning_rate, weight_decay = weight_decay)

    if use_cuda:

            model.to(device)
            criterion.to(device)

    for epoch_num in range(epochs):

            total_acc_train = 0
            total_loss_train = 0

            for train_input, train_label in tqdm(train_dataloader):

                train_label = train_label.to(device)
                mask = train_input['attention_mask'].squeeze(1).to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, train_label.long())
                total_loss_train += batch_loss.item()

                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()


            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():

                for val_input, val_label in val_dataloader:

                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].squeeze(1).to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)

                    output = model(input_id, mask)

                    batch_loss = criterion(output, val_label.long())
                    total_loss_val += batch_loss.item()

                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc

            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} | Train Accuracy: {total_acc_train / len(train_data): .3f} | Val Loss: {total_loss_val / len(val_data): .3f} | Val Accuracy: {total_acc_val / len(val_data): .3f}')


def evaluate(model, test_data):

    test = Dataset(test_data)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=32)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:

        model.to(device)

    total_acc_test = 0
    with torch.no_grad():

        for test_input, test_label in test_dataloader:

              test_label = test_label.to(device)
              mask = test_input['attention_mask'].squeeze(1).to(device)
              input_id = test_input['input_ids'].squeeze(1).to(device)

              output = model(input_id, mask)

              acc = (output.argmax(dim=1) == test_label).sum().item()
              total_acc_test += acc


    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')
    return total_acc_test / len(test_data)


def get_prob(model, data):
    train_data = Dataset(data)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model.to(device)

    probability = []
    correct_predictions = [] 

    with torch.no_grad():

        for train_input, train_label in train_dataloader:

            train_label = train_label.to(device)
            mask = train_input['attention_mask'].squeeze(1).to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            probabilities = F.softmax(output, dim=1)
            probability.extend(probabilities.detach().cpu().numpy())
            _, predicted_indices = torch.max(probabilities, dim=1)

            correct = (predicted_indices == train_label)
            correct_predictions.extend(correct.cpu().numpy())


    prob_one = np.array(probability)[:, 1]
    y_test = data['label']

    data['prob'] = prob_one
    data['correct'] = correct_predictions
    
    return data
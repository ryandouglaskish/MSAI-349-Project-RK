import numpy as np
import pandas as pd

import torch
torch.manual_seed(1)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn.functional as F

from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns

import time
from tqdm import tqdm


from helpers import *


class RaceDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    


    
class OneHiddenLayerReg(nn.Module):
    def __init__(self, input_size=240):
            super(OneHiddenLayerReg, self).__init__()
            self.fc1 = nn.Linear(input_size, 128) # 24 drivers x 10 laps
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 24)  # Output for 24 drivers

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Using raw scores for ranking
        return x

class TwoHiddenLayerReg(nn.Module):
    def __init__(self, input_size=240):
        super(TwoHiddenLayerReg, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32) 
        self.fc4 = nn.Linear(32, 24)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    
class OneHiddenLayerClassification(nn.Module):
    def __init__(self):
        super(OneHiddenLayerClassification, self).__init__()
        self.fc1 = nn.Linear(240, 128) # Input layer
        self.fc2 = nn.Linear(128, 64)  # Hidden layer 1
        self.fc3 = nn.Linear(64, 24)   # Output layer (24 positions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Activation function after hidden layer 1
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # No softmax here; nn.CrossEntropyLoss will apply it
        return x

class TwoHiddenLayerDropoutClassification(nn.Module):
    def __init__(self, input_size=240, dropout_rate=0.5):
        super(TwoHiddenLayerDropoutClassification, self).__init__()
        self.fc1 = nn.Linear(input_size, 128) 
        self.dropout1 = nn.Dropout(dropout_rate)  # Dropout layer after first fully connected layer
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(dropout_rate)  # Dropout layer after second fully connected layer
        self.fc3 = nn.Linear(64, 32)
        self.dropout3 = nn.Dropout(dropout_rate)  # Dropout layer after third fully connected layer
        self.fc4 = nn.Linear(32, 24)  # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)  # Applying dropout after ReLU activation
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)  # Applying dropout
        x = torch.relu(self.fc3(x))
        x = self.dropout3(x)  # Applying dropout
        x = self.fc4(x)  # No activation function here for classification
        return x

    

class WeightedRankingLoss(nn.Module):
    '''If winner wrong, penalize 5 points. If places 2-9 wrong, penalty is 1. If places 10-24 wrong, no penalty.'''
    def __init__(self, winner_weight=5.0):
        super(WeightedRankingLoss, self).__init__()
        self.winner_weight = winner_weight

    def forward(self, outputs, targets):
        loss = 0.0
        batch_size, num_drivers = outputs.shape
        for batch in range(batch_size):
            for i in range(num_drivers):
                for j in range(i + 1, num_drivers):
                    # Access the position of drivers i and j in the current batch
                    target_i = targets[batch, i].item()
                    target_j = targets[batch, j].item()

                    # Apply weights based on positions
                    weight = 1.0
                    if target_i == 1 or target_j == 1:
                        weight = self.winner_weight

                    elif target_i >= 10 and target_j >= 10:
                        weight = 0.0

                    # Calculate pairwise loss
                    predicted_diff = outputs[batch, i] - outputs[batch, j]
                    true_diff = torch.tensor(target_i - target_j, dtype=outputs.dtype, device=outputs.device)  # Convert true_diff to a tensor
                    loss += weight * F.relu(torch.sign(predicted_diff) * torch.sign(true_diff) - predicted_diff * true_diff)
                    loss += weight * F.relu(torch.sign(outputs[batch, i] - outputs[batch, j]) * torch.sign(torch.tensor(target_i - target_j, dtype=outputs.dtype, device=outputs.device)) - (outputs[batch, i] - outputs[batch, j]) * torch.tensor(target_i - target_j, dtype=outputs.dtype, device=outputs.device))
        return loss / (batch_size * num_drivers * (num_drivers - 1) / 2)
                  

def visualize_training(losses, path=None, save=True):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title("Training and Validation Loss")
    plt.plot(losses['training_losses'], label='Training Loss')
    plt.plot(losses['validation_losses'], label='Validation Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("1st Place Accuracy")
    plt.plot(losses['train_fp_acc'], label='Training Accuracy')
    plt.plot(losses['val_fp_acc'], label='Validation Accuracy')
    plt.legend()
    if save:
        plt.savefig(f'{path}Learning_Curve.png')
    else:
        plt.show()


def train_classification_model(path, train_loader, valid_loader, model, epochs=10, load=False, lr=0.001, viz=False):

    if not os.path.exists(path):
        os.makedirs(path)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    if not load:
        prev_epoch = 0
        performance = pd.DataFrame()
        print('New model created')
    else:
        checkpoint = torch.load(f'{path}model.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        prev_epoch = checkpoint['epoch']
        performance = checkpoint['performance']

    first_epoch = prev_epoch + 1
    last_epoch = epochs + first_epoch - 1

    training_losses = []
    validation_losses = []

    
    train_accs = []
    val_accs = []

    with tqdm(total=last_epoch, initial=first_epoch-1, desc='Training Progress', unit='epoch') as epoch_pbar:
        for epoch in range(first_epoch, last_epoch + 1):
            # Training Phase
            model.train()
            train_loss = 0.0

            # accuracy
            correct_preds = 0
            total_races = 0

            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                correct_preds += (predicted == targets).sum().item()
                total_races += inputs.size(0)

            average_train_loss = train_loss / len(train_loader)
            training_losses.append(average_train_loss)
            train_outputs = outputs

            # Calculate the accuracy
            train_acc = correct_preds / total_races
            train_accs.append(train_acc)

            # Validation Phase
            model.eval()

            val_loss = 0.0

            # accuracy
            correct_preds = 0
            total_races = 0

            with torch.no_grad():
                for inputs, targets in valid_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    correct_preds += (predicted == targets).sum().item()
                    total_races += inputs.size(0)

                average_val_loss = val_loss / len(valid_loader)
                validation_losses.append(average_val_loss)
                val_outputs = outputs

                # Calculate the accuracy
                val_acc = correct_preds / total_races
                val_accs.append(val_acc)
                
            epoch_pbar.set_description(f'Epoch {epoch}/{last_epoch} - Train Acc: {train_acc:.2f} | Val Acc: {val_acc:.2f}')
            epoch_pbar.update(1)

    performance = performance.append(pd.DataFrame({'epoch': list(range(first_epoch, last_epoch+1)), 'training_losses': training_losses, 'validation_losses': validation_losses, 'train_fp_acc': train_accs, 'val_fp_acc': val_accs}), ignore_index=True)


    if viz:
        visualize_training(performance, path, save=True)
    torch.save({
        'epoch': last_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'performance': performance,
    }, f'{path}model.pt')
    print('Model saved')
    notify('Training Complete', path.split('/')[-1])
    return performance



def train_regression_model(path, train_loader, valid_loader, model, criterion, epochs=10, load=False, lr=0.001, viz=False):

    if not os.path.exists(path):
        os.makedirs(path)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if not load:
        prev_epoch = 0
        performance = pd.DataFrame()
        print('New model created')
    else:
        checkpoint = torch.load(f'{path}model.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        prev_epoch = checkpoint['epoch']
        performance = checkpoint['performance']

    first_epoch = prev_epoch + 1
    last_epoch = epochs + first_epoch - 1

    training_losses = []
    validation_losses = []

    train_fp_accs = []
    val_fp_accs = []

    with tqdm(total=last_epoch, initial=first_epoch-1, desc='Training Progress', unit='epoch') as epoch_pbar:
        for epoch in range(first_epoch, last_epoch + 1):
            # Training Phase
            model.train()
            train_loss = 0.0
            # first place prediction accuracy
            correct_first_place_predictions = 0
            total_races = 0

            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                # Predicted and actual first-place finishers
                predicted_first_places = outputs.argmin(dim=1)
                actual_first_places = targets.argmin(dim=1)

                correct_first_place_predictions += (predicted_first_places == actual_first_places).sum().item()
                total_races += inputs.size(0)

            average_train_loss = train_loss / len(train_loader)
            training_losses.append(average_train_loss)
            train_outputs = outputs

            # Calculate the accuracy
            train_fp_acc = correct_first_place_predictions / total_races
            train_fp_accs.append(train_fp_acc)

            # Validation Phase
            model.eval()

            val_loss = 0.0

            # first place prediction accuracy
            correct_first_place_predictions = 0
            total_races = 0

            with torch.no_grad():
                for inputs, targets in valid_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()

                    # Predicted and actual first-place finishers
                    predicted_first_places = outputs.argmin(dim=1)
                    actual_first_places = targets.argmin(dim=1)

                    correct_first_place_predictions += (predicted_first_places == actual_first_places).sum().item()
                    total_races += inputs.size(0)

                average_val_loss = val_loss / len(valid_loader)
                validation_losses.append(average_val_loss)
                val_outputs = outputs

                # Calculate the accuracy
                val_fp_acc = correct_first_place_predictions / total_races
                val_fp_accs.append(val_fp_acc)
                
            epoch_pbar.set_description(f'Epoch {epoch}/{last_epoch} - Train Acc: {train_fp_acc:.2f} | Val Acc: {val_fp_acc:.2f}')
            epoch_pbar.update(1)

    performance = performance.append(pd.DataFrame({'epoch': list(range(first_epoch, last_epoch+1)), 'training_losses': training_losses, 'validation_losses': validation_losses, 'train_fp_acc': train_fp_accs, 'val_fp_acc': val_fp_accs}), ignore_index=True)


    if viz:
        visualize_training(performance, path, save=True)
    torch.save({
        'epoch': last_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'performance': performance,
    }, f'{path}model.pt')
    print('Model saved')
    notify('Training Complete', path.split('/')[-1])
    return performance

def get_model_performance(path):
    checkpoint = torch.load(f'{path}model.pt')
    performance = checkpoint['performance']
    return performance

def get_best_model_performance(path):
    checkpoint = torch.load(f'{path}model.pt')
    performance = checkpoint['performance']
    return performance.iloc[performance['val_fp_acc'].idxmax()]


def test_eval(path, test_loader, model, criterion):
    checkpoint = torch.load(f'{path}model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
 
    model.eval()

    test_loss = 0.0
    correct_first_place_predictions = 0
    total_races = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            # Predicted and actual first-place finishers
            predicted_first_places = outputs.argmin(dim=1)
            actual_first_places = targets.argmin(dim=1)

            correct_first_place_predictions += (predicted_first_places == actual_first_places).sum().item()
            total_races += inputs.size(0)

        average_test_loss = test_loss / len(test_loader)

        # Calculate the accuracy
        test_fp_acc = correct_first_place_predictions / total_races

    print(f'Loss: {average_test_loss:.4f} | Acc: {test_fp_acc:.6f}')
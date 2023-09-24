# Libraries for Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Libraries for Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Libraries for Time calculation
import time

# Libraries for numerical computation
import numpy as np
import scipy.stats as stats

# Libraries for handling datasets
import pandas as pd

# Libraries for Data Visualization
import matplotlib.pyplot as plt


def load_and_norm_data(name):
    '''
    Load data from a CSV file and returns the features and labels.
    The data is normalized using z-score normalization.
    '''
    # Load data
    data = pd.read_csv(name)

    # Define test and train group
    label = data['output']
    data.drop('output', axis=1, inplace=True)

    # Normelize data
    data = stats.zscore(data)
    return data, label


def ANN_classifier(data, label, num_of_ANN):
    '''
    Initializes, trains, and evaluates num_of_ANN ANNs and returns the best F1 Score.
    It uses a DataLoader to load the training and test datasets and uses BCEWithLogitsLoss as the loss function.
    '''
    start_time = time.time()
    # Batch Parameteres
    batchsizes = 20
    f1_ANN = 0

    Data_matrix = data.to_numpy()
    Label_matrix = label.to_numpy()

    # Convert into torch entiteis
    Data_matrix_T = torch.tensor(Data_matrix).float()
    Label_matrix_T = torch.tensor(Label_matrix).float()
    Label_matrix_T = Label_matrix_T[:, None]

    # Split data to test-train groups
    train_data, test_data, train_labels, test_labels = \
        train_test_split(Data_matrix_T, Label_matrix_T, test_size=0.2, random_state=33)

    # Define Data Set
    train_dataDataset = TensorDataset(train_data, train_labels)
    test_dataDataset = TensorDataset(test_data, test_labels)

    # Define the Loaders
    test_loader = DataLoader(test_dataDataset, batch_size=test_dataDataset.tensors[0].shape[0])
    train_loader = DataLoader(train_dataDataset, batch_size=batchsizes, shuffle=True, drop_last=True)

    for i in range(num_of_ANN):
        ANN_heart = ANN()
        trainAcc, testAcc, testLoss, trainLoss, best_f1_score = trainTheModel(ANN_heart, train_loader, test_loader)
        if best_f1_score > f1_ANN:
            f1_ANN = best_f1_score
    end_time = time.time()
    print("The ANN classifier took", end_time - start_time, "seconds to run.")
    return f1_ANN


def trainTheModel(ANN_heart, train_loader, test_loader):
    """
    Trains the given model using the Adam optimizer and BCEWithLogitsLoss.
    It returns the train and test accuracies and losses for each epoch, and the best F1 Score.
    """

    numepochs = 100
    best_accuracy = 0
    # loss function and optimizer
    lossfun = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(ANN_heart.parameters(), lr=.0001)

    # initialize losses and accuracies
    trainLoss = torch.zeros(numepochs)
    testLoss = torch.zeros(numepochs)
    trainAcc = torch.zeros(numepochs)
    testAcc = torch.zeros(numepochs)
    # loop over epochs
    for epochi in range(numepochs):

        # switch on training mode
        ANN_heart.train()

        # loop over training data batches
        batchAcc = []
        batchLoss = []
        for X, y in train_loader:
            # forward pass and loss
            yHat = ANN_heart(X)
            # yHat = yHat.squeeze(1)
            loss = lossfun(yHat, y)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # loss from this batch
            batchLoss.append(loss.item())

            # compute training accuracy for this batch
            batchAcc.append(100 * torch.mean(((yHat > 0) == y).float()).item())

        # now that we've trained through the batches, get their average training accuracy
        trainAcc[epochi] = np.mean(batchAcc)

        # and get average losses across the batches
        trainLoss[epochi] = np.mean(batchLoss)

        # test accuracy
        ANN_heart.eval()
        X, y = next(iter(test_loader))  # extract X,y from test dataloader
        with torch.no_grad():  # deactivates autograd
            yHat = ANN_heart(X)
        testAcc[epochi] = 100 * torch.mean(((yHat > 0) == y).float()).item()

        if testAcc[epochi] > best_accuracy:
            precision, recall, f1_score, support = precision_recall_fscore_support(y, (yHat > 0), average='weighted',
                                                                                   zero_division=0)
            best_f1_score = f1_score
        # test loss
        loss = lossfun(yHat, y)
        testLoss[epochi] = loss.item()

    return trainAcc, testAcc, testLoss, trainLoss, best_f1_score


class ANN(nn.Module):
    '''
    Defines a simple feed-forward neural network with two hidden layers and ReLU activation functions.
    '''

    def __init__(self):
        super().__init__()

        ### input layer
        self.input = nn.Linear(13, 32)

        ### hidden layers
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, 32)

        ### output layer
        self.output = nn.Linear(32, 1)

    # forward pass
    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.output(x)

def log_classifier(data, label):
    """
    This function trains a Logistic Regression model and returns the F1 Score of the model.
    """
    start_time = time.time()
    # Splitting the dataset into training and testing subsets
    train_data, test_data, train_labels, test_labels = train_test_split(data, label, test_size=0.2, random_state=33)

    # Initializing Logistic Regression model
    logmodel = LogisticRegression()

    # Training the model with the training dataset
    logmodel.fit(train_data, train_labels)

    # Making predictions on the test dataset
    predictions = logmodel.predict(test_data)

    # Calculating precision, recall, and F1 score of the model
    precision, recall, f1_score, support = precision_recall_fscore_support(test_labels, predictions,
                                                                               average='weighted')
    end_time = time.time()
    print("The Log classifier took", end_time - start_time, "seconds to run.")
    return f1_score  # Returning F1 score of the Logistic Regression model

def KNN_classifier(data, label):
    """
    This function trains K-Nearest Neighbors (KNN) classifiers with different K values and returns the best F1 Score.
    """
    start_time = time.time()
    best_f1_score = 0  # Initializing the variable to store the best F1 Score
    train_data, test_data, train_labels, test_labels = train_test_split(data, label, test_size=0.2, random_state=33)

    # Iterating over different K values to train multiple KNN models
    for i in range(1, 50):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(train_data, train_labels)
        predictions = knn.predict(test_data)

        # Calculating precision, recall, and F1 score of the model
        precision, recall, f1_score, support = precision_recall_fscore_support(test_labels, predictions,
                                                                                   average='weighted')
         # Updating the best F1 Score if the current model's F1 Score is better
        if f1_score > best_f1_score:
         best_f1_score = f1_score

    end_time = time.time()
    print("The KNN classifier took", end_time - start_time, "seconds to run.")
    return best_f1_score  # Returning the best F1 Score among all KNN models trained

def RF_classifier(data, label):
    """
    This function trains Random Forest classifiers with different numbers of trees and returns the best F1 Score.
    """
    start_time = time.time()
    best_f1_score = 0  # Initializing the variable to store the best F1 Score
    train_data, test_data, train_labels, test_labels = train_test_split(data, label, test_size=0.2, random_state=33)

    # Iterating over different number of trees to train multiple Random Forest models
    for num_of_tress in np.linspace(100, 1000, 10):
        rfc = RandomForestClassifier(n_estimators=int(num_of_tress))
        rfc.fit(train_data, train_labels)
        predictions = rfc.predict(test_data)

        # Calculating precision, recall, and F1 score of the model
        precision, recall, f1_score, support = precision_recall_fscore_support(test_labels, predictions,
                                                                                   average='weighted')

        # Updating the best F1 Score if the current model's F1 Score is better
        if f1_score > best_f1_score:
            best_f1_score = f1_score
    end_time = time.time()
    print("The RF classifier took", end_time - start_time, "seconds to run.")
    return best_f1_score  # Returning the best F1 Score among all Random Forest models trained

def SVM_classifier(data, label):
    """
    This function performs Grid Search to find the optimal parameters for the SVM classifier and returns the F1 Score.
    """
    start_time = time.time()
    # Splitting the dataset into training and testing subsets
    train_data, test_data, train_labels, test_labels = train_test_split(data, label, test_size=0.2, random_state=33)

    # Defining parameter grid for Grid Search
    param_grid = {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'degree': [2, 3, 4, 5],  # Only used by 'poly' kernel
        'class_weight': [None, 'balanced']
        }

    # Initializing Grid Search for SVM with the specified parameter grid
    grid = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(train_data, train_labels)

    # Making predictions using the best estimator found by Grid Search
    predictions = grid.predict(test_data)

    # Calculating precision, recall, and F1 score of the model
    precision, recall, f1_score, support = precision_recall_fscore_support(test_labels, predictions,
                                                                               average='weighted')
    end_time = time.time()
    print("The SVM classifier took", end_time - start_time, "seconds to run.")
    return f1_score  # Returning F1 score of the optimized SVM model


def classfier_bar_plot(f1_log, f1_KNN, f1_RF, f1_ANN, f1_SVM):
    # Plots the F1 Scores of all classifiers in a bar plot for visual comparison.

    classifiers = ['Logistic Regression', 'KNN', 'Random Forest', 'ANN', 'SVM']
    f1_scores = [f1_log, f1_KNN, f1_RF, f1_ANN, f1_SVM]  # Replace with actual F1 scores

    plt.figure(figsize=(10, 6))
    plt.bar(classifiers, f1_scores, color=['blue', 'green', 'red', 'yellow'])
    plt.ylabel('F1 Score')
    plt.title('F1 Score of Different Classifiers')
    plt.ylim(0.5, 1)
    plt.show()

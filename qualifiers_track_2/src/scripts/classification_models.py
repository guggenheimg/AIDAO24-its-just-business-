import numpy as np
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


class LogRegPCA:
    def __init__(self, pca=True):
        self.pca = PCA() if pca else None
        self.model = LogisticRegression()

    def model_training(self, x, y):
        x = self.preprocess(x)

        if self.pca is not None:
            x = self.pca.fit_transform(x)

        self.model.fit(x, y)

        acc = self.model.score(x, y)
        print('Accuracy on train:', round(acc, 3))

        return acc

    def model_predict(self, x):
        x = self.preprocess(x)

        if self.pca is not None:
            x = self.pca.transform(x)

        y_pred = self.model.predict(x)
        return y_pred

    def model_testing(self, x, y):
        y_pred = self.model_predict(x)

        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        print('Accuracy on test:', round(acc, 3))
        print('F1 score on test:', round(f1, 3))
        cm = confusion_matrix(y, y_pred)

        return cm, acc, f1

    def preprocess(self, x):
        vecs = zscore(x, axis=0)

        for i in vecs:
            np.fill_diagonal(i, 0)

        vecs = vecs.reshape((x.shape[0], -1))

        return vecs



import lightgbm as lgb
import pickle
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
    
class LightGBMModel:
    def __init__(self, params=None):
        if params is None:
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'num_leaves': 40,
                'max_depth': 6,
                'learning_rate': 0.037091216078881495,
                'min_data_in_leaf': 32,
                'bagging_fraction': 0.6982271066754833,
                'bagging_freq': 5,
                'feature_fraction': 0.6995573494168418,
                'boosting_type': 'gbdt',
                'max_bin': 255,
                'n_jobs': -1,
                'is_unbalance': True, 
                'lambda_l1': 0.45035216849608783,  
                'lambda_l2': 0.7080197077200181  
            }
        self.params = params
        self.model = None
    
    def model_training(self, X_train, y_train, X_val, y_val, use_cv=False):
        X_train = self.preprocess(X_train)
        X_val = self.preprocess(X_val)
        
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        if use_cv:
            cv_results = lgb.cv(
                self.params,
                train_data,
                num_boost_round=3000,
                nfold=5,
                stratified=True,
                early_stopping_rounds=100,
                verbose_eval=100
            )

            best_num_boost_round = len(cv_results['binary_logloss-mean'])
            print(f"Best number of iterations: {best_num_boost_round}")

            self.model = lgb.train(
                self.params,
                train_data,
                num_boost_round=best_num_boost_round,
                valid_sets=[valid_data],
                callbacks=[lgb.early_stopping(stopping_rounds=100)]
            )
        else:
            self.model = lgb.train(
                self.params, 
                train_data, 
                num_boost_round=3000, 
                valid_sets=[valid_data], 
                callbacks=[lgb.early_stopping(stopping_rounds=100)]
            )
        
        y_pred_train = self.model.predict(X_train)
        y_pred_train_binary = (y_pred_train > 0.5).astype(int)
        train_acc = accuracy_score(y_train, y_pred_train_binary)
        return train_acc
    
    def model_testing(self, X_test, y_test):
        X_test = self.preprocess(X_test)
        
        y_pred = self.model.predict(X_test)
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        acc = accuracy_score(y_test, y_pred_binary)
        f1 = f1_score(y_test, y_pred_binary)
        
        return acc, f1
    
    def model_predict(self, X):
        X = self.preprocess(X)
        y_pred = self.model.predict(X)
        return (y_pred > 0.5).astype(int)
    
    def preprocess(self, x):
        vecs = zscore(x, axis=0)

        for i in vecs:
            np.fill_diagonal(i, 0)

        vecs = vecs.reshape((x.shape[0], -1))

        return vecs




import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score

class PyTorchNN:
    def __init__(self, input_size, learning_rate=0.001, batch_size=32, epochs=50):
        self.model = self.SimpleNN(input_size)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.batch_size = batch_size
        self.epochs = epochs

    class SimpleNN(nn.Module):
        def __init__(self, input_size):
            super(PyTorchNN.SimpleNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(32 * 209 * 209, 64)
            self.fc2 = nn.Linear(64, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = x.unsqueeze(1)
            x = self.conv1(x)
            x = self.relu(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.sigmoid(x)
            return x

    def model_training(self, X_train, y_train, X_val, y_val):
        for epoch in range(self.epochs):
            self.model.train()
            for i in range(0, len(X_train), self.batch_size):
                X_batch = X_train[i:i+self.batch_size]
                y_batch = y_train[i:i+self.batch_size].view(-1, 1)

                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()

            print(f'Epoch {epoch+1}/{self.epochs}, Loss: {loss.item()}')

    def model_testing(self, X_val, y_val):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_val)
            predicted = (outputs > 0.5).float()

            predicted = predicted.int().view(-1)
            y_val = y_val.int().view(-1)

            print(f'Размер y_val: {y_val.shape}, Размер predicted: {predicted.shape}')
            
            acc = accuracy_score(y_val, predicted)
            f1 = f1_score(y_val, predicted)
        
        return acc, f1



    def model_predict(self, X):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
        return (outputs > 0.5).float()


    
    def preprocess(self, x):
        vecs = zscore(x, axis=0)

        for i in vecs:
            np.fill_diagonal(i, 0)

        vecs = vecs.reshape((x.shape[0], -1))

        return vecs
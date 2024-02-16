

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import json
writer = SummaryWriter("./result/dnm_loss")


class dnm_trainer():
    def __init__(self, model, train_dataloader, val_dataloader, num_epochs=500, learning_rate=1e-3, num=0, repeat=0):
        super().__init__()
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.num_epochs = num_epochs
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(model.parameters(), lr = learning_rate)
        self.early_stopping = 0
        self.patience = 5
        self.best_loss = float('inf')
        self.num = num
        self.repeat = repeat
        self.lr = learning_rate
        self.best_params = {}
    def train(self):
        for epoch in range(self.num_epochs):
            # train
            self.model.train()
            total_train_loss = 0
            for inputs, labels in tqdm(self.train_dataloader, desc=f"Epoch: {epoch}/{self.num_epochs}"):
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_train_loss += loss.item()
                loss.backward()

                # writer.add_histogram(f'Gradients/fc1', self.model.fc1.weight.grad, epoch)
                # writer.add_histogram(f'Gradients/synapticLayer_weight', self.model.synapticLayer.weight.grad, epoch)
                # writer.add_histogram(f'Gradients/synapticLayer_theta', self.model.synapticLayer.theta.grad, epoch)
                # writer.add_histogram(f'Gradients/membraneLayer_weight', self.model.membraneLayer.weight.grad, epoch)

                self.optimizer.step()
            avg_train_loss = total_train_loss / len(self.train_dataloader)
            writer.add_scalar(f'{self.repeat}_{self.num}_Training loss', avg_train_loss, epoch)
        
            # val
            self.model.eval()
            total_val_loss = 0
            with torch.no_grad():  # no grad
                for inputs, labels in self.val_dataloader:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    total_val_loss += loss.item()
            avg_val_loss = total_val_loss / len(self.val_dataloader)
            writer.add_scalar(f'{self.repeat}_{self.num}_Validation loss', avg_val_loss, epoch)

            # early stopping
            if avg_val_loss < self.best_loss:
                self.best_loss = avg_val_loss
                self.best_params = {'dendriteNum': self.model.dendriteNum,
                               'k': self.model.k,
                               'ks': self.model.ks,
                               'thetas': self.model.thetas,
                               'lr': self.lr}
                torch.save(self.model.state_dict(), f'./result/modelBest/dnm_{self.num}_modelBestParameters')
                self.early_stopping = 0
            else:
                self.early_stopping += 1
                if self.early_stopping >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break  
            with open(f'./result/Hyperparameter/dnm_{self.num}_best_hyperparams.json', 'w') as f:
                json.dump(self.best_params, f)
        writer.close()






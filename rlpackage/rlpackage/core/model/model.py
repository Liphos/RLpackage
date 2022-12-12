import torch.nn as F
import torch
from typing import Tuple

class DQNModel(F.Module):
    def __init__(self, observation_shape:int, action_shape:int):
        super(DQNModel, self).__init__()
        self.layers = []
        self.dense1 = F.Linear(observation_shape, 64)
        self.layers.append(self.dense1)
        self.dense2 = F.Linear(64, action_shape)
        self.layers.append(self.dense2)
        self.activation = F.ReLU()
        
    def forward(self, x:torch.Tensor):
        x = self.activation(self.dense1(x))
        x = self.dense2(x)
        return x
    
    def save_txt(self, filename:str):
        """Save the layers in a txt

        Args:
            filename (str): path to the txt file
        """
        with open(filename, 'w') as f:
            for layer in self.layers:
                f.write(str(layer._get_name) + "\n")
        f.close()
        
class PPOPolicy(F.Module):
    def __init__(self, observation_shape:int, action_shape:int):
        super().__init__()
        self.layers = []
        self.dense1 = F.Linear(observation_shape, 64)
        self.layers.append(self.dense1)
        self.dense2_mean = F.Linear(64, action_shape)
        self.dense2_std = F.Linear(64, action_shape)
        self.layers.append(self.dense2_mean)
        self.layers.append(self.dense2_std)
        self.activation = F.ReLU()
        self.std_activation = torch.exp
        
    def forward(self, x:torch.Tensor):
        x = self.activation(self.dense1(x))
        means, stds = self.dense2_mean(x), self.std_activation(self.dense2_std(x))
        return means, stds
    
    def save_txt(self, filename:str):
        """Save the layers in a txt

        Args:
            filename (str): path to the txt file
        """
        with open(filename, 'w') as f:
            for layer in self.layers:
                f.write(str(layer._get_name) + "\n")
        f.close()

class PPOCritic(F.Module):
    def __init__(self, observation_shape:int, action_shape:int):
        super().__init__()
        self.layers = []
        self.dense1 = F.Linear(observation_shape, 64)
        self.layers.append(self.dense1)
        self.dense2 = F.Linear(64, action_shape)
        self.layers.append(self.dense2)
        self.activation = F.ReLU()
        
    def forward(self, x:torch.Tensor):
        x = self.activation(self.dense1(x))
        x = self.dense2(x)
        return x
    
    def save_txt(self, filename:str):
        """Save the layers in a txt

        Args:
            filename (str): path to the txt file
        """
        with open(filename, 'w') as f:
            for layer in self.layers:
                f.write(str(layer._get_name) + "\n")
        f.close()
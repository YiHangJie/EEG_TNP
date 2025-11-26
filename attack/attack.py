from abc import ABC, abstractmethod
import torch

class Attack(torch.nn.Module, ABC):

    def __init__(self, model, device, n_classes=10):
        super(Attack, self).__init__()
        self.model = model.eval()
        self.device = device
        self.n_classes = n_classes

    @abstractmethod
    def forward(self, x, y):
        raise NotImplementedError
    
    def get_std_per_channel(self, x):
        return torch.std(x, dim=-1)
import torch

from attack.attack import Attack

class FGSM(Attack):
    def __init__(self, model, device='cuda', eps=8/255, n_classes=10):
        super().__init__(model, device)
        self.eps = eps

    def forward(self, x, y):
        x = x.clone().detach().to(self.device)
        y = y.clone().detach().to(self.device)
        x.requires_grad = True

        loss_fn = torch.nn.CrossEntropyLoss()

        output = self.model(x)
        loss = loss_fn(output, y)
        loss.backward()

        x_adv = x + self.eps * torch.sign(x.grad.detach())

        return x_adv.detach()
    

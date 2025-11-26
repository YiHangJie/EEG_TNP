import torch
import torchattacks

from attack.attack import Attack

class PGD(Attack):
    def __init__(self, model, device="cuda", eps=8/255, alpha=1/255, steps=100, n_classes=10):
        super().__init__(model, device)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps

    def forward(self, x, y):
        x = x.clone().detach().to(self.device)
        y = y.clone().detach().to(self.device)
        x_adv = x.clone().detach().to(self.device)
        x_adv.requires_grad = True

        loss_fn = torch.nn.CrossEntropyLoss()

        for i in range(self.steps):
            x_adv.requires_grad = True
            output = self.model(x_adv)
            loss = loss_fn(output, y)
            loss.backward()

            x_adv = x_adv + self.alpha * torch.sign(x_adv.grad.detach())
            delta = torch.clamp(x_adv - x, -self.eps, self.eps)
            x_adv = (x + delta).detach()

        return x_adv.detach()
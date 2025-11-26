import torch
import torchattacks

from attack.attack import Attack

class CW(Attack):
    def __init__(self, model, device="cuda", lr=0.1, steps=100, n_classes=10):
        super(CW, self).__init__(model, device)
        self.lr = lr
        self.steps = steps
        self.c = 10
        self.kappa = 0
    
    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # w = torch.zeros_like(images).detach() # Requires 2x times
        # w = self.inverse_tanh_space(images).detach()
        w = images.clone().detach()
        w.requires_grad = True

        best_adv_images = images.clone().detach()
        best_L2 = 1e10 * torch.ones((len(images))).to(self.device)
        prev_cost = 1e10
        dim = len(images.shape)

        MSELoss = torch.nn.MSELoss(reduction="none")
        Flatten = torch.nn.Flatten()

        optimizer = torch.optim.Adam([w], lr=self.lr)

        for step in range(self.steps):
            # Get adversarial images
            # adv_images = self.tanh_space(w)
            adv_images = w

            # Calculate loss
            current_L2 = MSELoss(Flatten(adv_images), Flatten(images)).sum(dim=1)
            # current_L2 = MSELoss(adv_images, images)
            L2_loss = current_L2.sum()

            outputs = self.model(adv_images)
            f_loss = self.f(outputs, labels).sum()

            cost = L2_loss + self.c * f_loss

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # Update adversarial images
            pre = torch.argmax(outputs.detach(), 1)
            # If the attack is not targeted we simply make these two values unequal
            condition = (pre != labels).float()

            # Filter out images that get either correct predictions or non-decreasing loss,
            # i.e., only images that are both misclassified and loss-decreasing are left
            mask = condition * (best_L2 > current_L2.detach())
            best_L2 = mask * current_L2.detach() + (1 - mask) * best_L2

            mask = mask.view([-1] + [1] * (dim - 1))
            best_adv_images = mask * adv_images.detach() + (1 - mask) * best_adv_images

            # Early stop when loss does not converge.
            # max(.,1) To prevent MODULO BY ZERO error in the next step.
            if step % max(self.steps // 10, 1) == 0:
                if cost.item() > prev_cost:
                    return best_adv_images
                prev_cost = cost.item()

        return best_adv_images
    
    # def tanh_space(self, x):
    #     return 1 / 2 * (torch.tanh(x) + 1)
    
    # def inverse_tanh_space(self, x):
    #     # torch.atanh is only for torch >= 1.7.0
    #     # atanh is defined in the range -1 to 1
    #     return self.atanh(torch.clamp(x * 2 - 1, min=-1, max=1))

    # def atanh(self, x):
    #     return 0.5 * torch.log((1 + x) / (1 - x))

    # f-function in the paper
    def f(self, outputs, labels):
        one_hot_labels = torch.eye(outputs.shape[1]).to(self.device)[labels]

        # find the max logit other than the target class
        other = torch.max((1 - one_hot_labels) * outputs, dim=1)[0]
        # get the target class's logit
        real = torch.max(one_hot_labels * outputs, dim=1)[0]

        return torch.clamp((real - other), min=-self.kappa)
import torchattacks

from torchattacks.wrappers.multiattack import MultiAttack
from attack.attack import Attack
from attack.apgd import APGD
from attack.apgdt import APGDT
from attack.fab import FAB
from attack.square import Square

class AutoAttack(Attack):
    def __init__(self, model, device='cuda', norm='Linf', eps=8/255, version='standard', seed=42, n_classes=10, verbose=False):
        super().__init__(model, device, n_classes)
        self.norm = norm
        self.eps = eps
        self.version = version
        self.seed = seed
        self.verbose = verbose

        if self.version == 'rand':
            self._autoattack = MultiAttack(
                [
                    APGD(
                        model,
                        eps=eps,
                        norm=norm,
                        seed=self.seed,
                        verbose=self.verbose,
                        loss="ce",
                        eot_iter=20,
                        n_restarts=1,
                    ),
                    APGD(
                        model,
                        eps=eps,
                        norm=norm,
                        seed=self.seed,
                        verbose=self.verbose,
                        loss="dlr",
                        eot_iter=20,
                        n_restarts=1,
                    ),
                ]
            )
        elif version == "standard":  # ['apgd-ce', 'apgd-t', 'fab-t', 'square']
            self._autoattack = MultiAttack(
                [
                    APGD(
                        model,
                        eps=eps,
                        norm=norm,
                        seed=self.seed,
                        verbose=self.verbose,
                        loss="ce",
                        n_restarts=1,
                    ),
                    APGDT(
                        model,
                        eps=eps,
                        norm=norm,
                        seed=self.seed,
                        verbose=self.verbose,
                        n_classes=n_classes,
                        n_restarts=1,
                    ),
                    FAB(
                        model,
                        eps=eps,
                        norm=norm,
                        seed=self.seed,
                        verbose=self.verbose,
                        multi_targeted=True,
                        n_classes=n_classes,
                        n_restarts=1,
                    ),
                    Square(
                        model,
                        eps=eps,
                        norm=norm,
                        seed=self.seed,
                        verbose=self.verbose,
                        n_queries=5000,
                        n_restarts=1,
                    ),
                ]
            )

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv_images = self._autoattack(images, labels)

        return adv_images
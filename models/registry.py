from torcheeg.models import ATCNet, Conformer, EEGNet, TCNet, TSCeption

from models.deepconvnet import DeepConvNet


class ProjectTCNet(TCNet):
    """TCNet wrapper for this project's `[B, 1, C, T]` EEG batches."""

    def forward(self, x):
        if x.dim() == 4 and x.size(1) == 1:
            x = x.squeeze(1)
        return super().forward(x)


MODEL_CLASSES = {
    "eegnet": EEGNet,
    "tsception": TSCeption,
    "atcnet": ATCNet,
    "conformer": Conformer,
    "tcnet": ProjectTCNet,
    "deepconvnet": DeepConvNet,
}

MODEL_CHOICES = tuple(MODEL_CLASSES.keys())

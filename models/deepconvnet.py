import torch
from torch import nn
from torch.nn import init


class DeepConvNet(nn.Module):
    """Deep ConvNet backbone adapted from Braindecode Deep4Net.

    输入保持本项目/TorchEEG 的 `[batch, channels, time]` 约定；内部转换为
    `[batch, 1, time, channels]` 后执行 temporal conv + spatial conv，再接三层
    temporal convolution/pooling block。
    """

    def __init__(
        self,
        num_electrodes,
        num_classes,
        chunk_size,
        final_conv_length="auto",
        n_filters_time=25,
        n_filters_spat=25,
        filter_time_length=10,
        pool_time_length=3,
        pool_time_stride=3,
        n_filters_2=50,
        filter_length_2=10,
        n_filters_3=100,
        filter_length_3=10,
        n_filters_4=200,
        filter_length_4=10,
        drop_prob=0.5,
        batch_norm=True,
        batch_norm_alpha=0.1,
    ):
        super().__init__()
        self.num_electrodes = num_electrodes
        self.num_classes = num_classes
        self.chunk_size = chunk_size
        self.final_conv_length = final_conv_length
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha

        self.conv_time_spat = nn.Sequential()
        self.conv_time_spat.add_module(
            "conv_time",
            nn.Conv2d(1, n_filters_time, (filter_time_length, 1), bias=True),
        )
        self.conv_time_spat.add_module(
            "conv_spat",
            nn.Conv2d(
                n_filters_time,
                n_filters_spat,
                (1, num_electrodes),
                bias=not batch_norm,
            ),
        )
        n_filters_conv = n_filters_spat

        if batch_norm:
            self.bnorm = nn.BatchNorm2d(
                n_filters_conv, momentum=batch_norm_alpha, affine=True, eps=1e-5
            )
        else:
            self.bnorm = nn.Identity()
        self.conv_nonlin = nn.ELU()
        self.pool = nn.MaxPool2d(
            kernel_size=(pool_time_length, 1), stride=(pool_time_stride, 1)
        )
        self.pool_nonlin = nn.Identity()

        self.drop_2 = nn.Dropout(p=drop_prob)
        self.conv_2 = nn.Conv2d(
            n_filters_conv, n_filters_2, (filter_length_2, 1), bias=not batch_norm
        )
        self.bnorm_2 = (
            nn.BatchNorm2d(n_filters_2, momentum=batch_norm_alpha, affine=True, eps=1e-5)
            if batch_norm
            else nn.Identity()
        )
        self.nonlin_2 = nn.ELU()
        self.pool_2 = nn.MaxPool2d(
            kernel_size=(pool_time_length, 1), stride=(pool_time_stride, 1)
        )
        self.pool_nonlin_2 = nn.Identity()

        self.drop_3 = nn.Dropout(p=drop_prob)
        self.conv_3 = nn.Conv2d(
            n_filters_2, n_filters_3, (filter_length_3, 1), bias=not batch_norm
        )
        self.bnorm_3 = (
            nn.BatchNorm2d(n_filters_3, momentum=batch_norm_alpha, affine=True, eps=1e-5)
            if batch_norm
            else nn.Identity()
        )
        self.nonlin_3 = nn.ELU()
        self.pool_3 = nn.MaxPool2d(
            kernel_size=(pool_time_length, 1), stride=(pool_time_stride, 1)
        )
        self.pool_nonlin_3 = nn.Identity()

        self.drop_4 = nn.Dropout(p=drop_prob)
        self.conv_4 = nn.Conv2d(
            n_filters_3, n_filters_4, (filter_length_4, 1), bias=not batch_norm
        )
        self.bnorm_4 = (
            nn.BatchNorm2d(n_filters_4, momentum=batch_norm_alpha, affine=True, eps=1e-5)
            if batch_norm
            else nn.Identity()
        )
        self.nonlin_4 = nn.ELU()
        self.pool_4 = nn.MaxPool2d(
            kernel_size=(pool_time_length, 1), stride=(pool_time_stride, 1)
        )
        self.pool_nonlin_4 = nn.Identity()

        if final_conv_length == "auto":
            final_conv_length = self._infer_final_conv_length()
        self.final_layer = nn.Sequential()
        self.final_layer.add_module(
            "conv_classifier",
            nn.Conv2d(n_filters_4, num_classes, (final_conv_length, 1), bias=True),
        )
        self.final_layer.add_module("squeeze", _SqueezeFinalOutput())
        self._initialize_weights()

    def _features(self, x):
        if x.dim() == 4 and x.size(1) == 1:
            x = x.squeeze(1)
        if x.dim() != 3:
            raise ValueError(f"DeepConvNet expects [batch, channels, time], got {x.shape}")
        x = x.unsqueeze(1).permute(0, 1, 3, 2)
        x = self.conv_time_spat(x)
        x = self.bnorm(x)
        x = self.conv_nonlin(x)
        x = self.pool_nonlin(self.pool(x))

        x = self.drop_2(x)
        x = self.pool_nonlin_2(self.pool_2(self.nonlin_2(self.bnorm_2(self.conv_2(x)))))
        x = self.drop_3(x)
        x = self.pool_nonlin_3(self.pool_3(self.nonlin_3(self.bnorm_3(self.conv_3(x)))))
        x = self.drop_4(x)
        x = self.pool_nonlin_4(self.pool_4(self.nonlin_4(self.bnorm_4(self.conv_4(x)))))
        return x

    def _infer_final_conv_length(self):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, self.num_electrodes, self.chunk_size)
            length = self._features(dummy).shape[2]
        self.train(was_training)
        return int(length)

    def _initialize_weights(self):
        init.xavier_uniform_(self.conv_time_spat.conv_time.weight, gain=1)
        init.constant_(self.conv_time_spat.conv_time.bias, 0)
        init.xavier_uniform_(self.conv_time_spat.conv_spat.weight, gain=1)
        if self.conv_time_spat.conv_spat.bias is not None:
            init.constant_(self.conv_time_spat.conv_spat.bias, 0)
        for name in ("bnorm", "bnorm_2", "bnorm_3", "bnorm_4"):
            module = getattr(self, name)
            if isinstance(module, nn.BatchNorm2d):
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)
        for name in ("conv_2", "conv_3", "conv_4"):
            module = getattr(self, name)
            init.xavier_uniform_(module.weight, gain=1)
            if module.bias is not None:
                init.constant_(module.bias, 0)
        init.xavier_uniform_(self.final_layer.conv_classifier.weight, gain=1)
        init.constant_(self.final_layer.conv_classifier.bias, 0)

    def forward(self, x):
        x = self._features(x)
        return self.final_layer(x)


class _SqueezeFinalOutput(nn.Module):
    def forward(self, x):
        return x.squeeze(3).squeeze(2)

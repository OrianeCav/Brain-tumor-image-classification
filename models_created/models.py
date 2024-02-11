import torch.nn as nn
import torch.nn.functional as F


class classifier_large(nn.Module):
    def __init__(
        self, conv_kernel_number, conv_kernel_size, maxpool_kernel_size, dropout_rate
    ):
        super(classifier_large, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=conv_kernel_number,
            kernel_size=conv_kernel_size,
            padding=int((conv_kernel_size - 1) // 2),
        )
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(
            in_features=conv_kernel_number * (150 // maxpool_kernel_size) ** 2,
            out_features=2,
        )
        self.maxpool_kernel_size = maxpool_kernel_size

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=self.maxpool_kernel_size, padding=0)
        x = self.dropout1(x)

        flat = nn.Flatten(1, 3)
        x = flat(x)
        x = self.fc(x)
        x = F.log_softmax(x, 1)

        return x

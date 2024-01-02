from torch import nn


class MyAwesomeModel(nn.Module):
    """
    An awesome small convolutional model for classification 
    on a corrupted version of the MNIST dataset.
    Images shape: 1 x 28 x 28
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 7 * 7, 32)
        self.fc2 = nn.Linear(32, 10)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 7 * 7)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.log_softmax(self.fc2(x))
        return x
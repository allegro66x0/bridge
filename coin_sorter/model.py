import torch.nn as nn

IMG_SIZE = 64

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * (IMG_SIZE // 4) * (IMG_SIZE // 4), 128)
        self.fc2 = nn.Linear(128, 2) # Binary classification
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * (IMG_SIZE // 4) * (IMG_SIZE // 4))
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

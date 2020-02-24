import torch.nn as nn
import torch.nn.functional as F

from .prune import PruningModule, MaskedLinear

class LeNet(PruningModule):
    def __init__(self, mask=False):
        super(LeNet, self).__init__()
        linear = MaskedLinear if mask else nn.Linear
        self.fc1 = linear(784, 128)
        self.fc2 = linear(128, 64)
        # self.fc4 = linear(64,32)
        self.fc3 = linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc4(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x

class LeNet_3(PruningModule):
    def __init__(self, mask=False):
        super(LeNet_3, self).__init__()
        linear = MaskedLinear if mask else nn.Linear
        self.fc1 = linear(784, 128)
        self.fc2 = linear(128, 64)
        self.fc4 = linear(64,32)
        self.fc3 = linear(32, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc4(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x


class ConvNet(PruningModule):
    def __init__(self,mask=False):
        super(ConvNet, self).__init__()
        linear = MaskedLinear if mask else nn.Linear
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = linear(7 * 7 * 64, 1000)
        self.fc2 = linear(1000, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = F.log_softmax(self.fc2(out), dim=1)
        return out

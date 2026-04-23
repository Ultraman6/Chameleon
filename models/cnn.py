import torch

class Model(torch.nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.ReLU()
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.ReLU()
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(64*5*5, 384),
            torch.nn.ReLU(),
            torch.nn.Linear(384, 192),
            torch.nn.ReLU(),
            torch.nn.Linear(192, num_classes)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = torch.reshape(x, (x.shape[0], -1))
        x = self.classifier(x)
        return x

def CNN(num_classes=10):
    return Model(num_classes=num_classes)
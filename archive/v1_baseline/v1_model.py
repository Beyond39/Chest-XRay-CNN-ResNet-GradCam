"""
这一个版本是baseline的版本，我只在这一版本里处理了训练集，进行了epoch = 5轮训练，并且得到了还不错的训练结果。
这里也有最简单的SimpleCNN的处理，看起来还是不错的，我似乎可以处理一定的工程量了。
我们有3个卷积层，进行线性连接之后还是可以使用的。
我觉得这是最基础的一版了，但是我们配置好了环境，处理了最简单的matplotlib的图像，这让我感觉我还是可以做出一些东西的。

"""
import torch 
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN,self).__init__()

        #使用卷积处理数据
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # 现在特征图尺寸：
        # 通道数 64
        # 空间尺寸 28 x 28
        # 所以 flatten 后是 64 * 28 * 28

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(64 * 28 * 28 , 256),
            nn.ReLU(),

            nn.Linear(256 ,128),
            nn.ReLU(),

            nn.Linear(128, num_classes)   
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

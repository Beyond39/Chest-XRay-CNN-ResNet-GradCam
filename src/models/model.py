# model.py
import torch
import torch.nn as nn
import torchvision.models as models

class ChestResNet18(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        """
        :param num_classes: 分类数，正常 vs 肺炎
        :param pretrained: 是否使用迁移学习（ImageNet预训练权重）。
        """
        super(ChestResNet18, self).__init__()
        
        if pretrained:
            # 采用预训练权重（迁移学习）。相当于这个模型已经看过了上百万张猫狗飞机的图片，
            # 它已经学会了如何提取“边缘、纹理、轮廓”，我们只需要让它适应 X 光片即可。
            weights = models.ResNet18_Weights.DEFAULT
            self.resnet = models.resnet18(weights=weights)
            print("=> 已加载 ResNet-18 预训练权重！(Transfer Learning 开启)")
        else:
            # 随机初始化权重。如果你想单纯对比架构优势（A/B Test），可以设为 False。
            self.resnet = models.resnet18(weights=None)
            print("=> 未加载预训练权重，模型将从头开始训练 (Training from scratch)。")

        # 2. 改造“分类头” (Head)
        # 官方的 ResNet18 是为 ImageNet 设计的，最后输出是 1000 个类别。
        # 我们需要把它截断，换成我们自己的 2分类。
        
        # 获取原模型全连接层 (fc) 的输入特征数 (ResNet18 这里固定是 512)
        in_features = self.resnet.fc.in_features 
        
        # 替换官方的全连接层。
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        # 前向传播极其优雅，所有的残差连接、激活、池化都在 self.resnet 内部完成了
        return self.resnet(x)

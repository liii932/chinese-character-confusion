import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, ResNet

def resnet18_gray_64(num_classes: int = 81):
    """
    适配64×64灰度输入的ResNet18
    参考项目设计书6.1
    """
    model = ResNet(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=num_classes)

    # 替换第一层：灰度输入，3×3卷积，stride=1，不去做maxpool
    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )
    model.bn1 = nn.BatchNorm2d(64)
    # 移除原始的maxpool！64分辨率不需要
    model.maxpool = nn.Identity()

    return model


if __name__ == "__main__":
    # 自测：给一个随机 [1,1,64,64] 的灰度输入，看输出shape
    net = resnet18_gray_64(num_classes=81)
    x = torch.randn(1, 1, 64, 64)
    out = net(x)
    print(f"输入shape {tuple(x.shape)}")
    print(f"输出logits shape {tuple(out.shape)}")
    print(f"输出类别数：{out.size(-1)}")

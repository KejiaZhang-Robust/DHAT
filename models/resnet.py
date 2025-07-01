import torch

from .utils import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class proj_head(nn.Module):
    def __init__(self, ch):
        super(proj_head, self).__init__()
        self.in_features = ch
        ch1 = ch//2
        ch2 = ch//4

        self.fc1 = nn.Linear(ch, ch1)
        self.bn1 = nn.BatchNorm1d(ch1)
        self.fc2 = nn.Linear(ch1, ch2, bias=False)
        self.bn2 = nn.BatchNorm1d(ch2)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)

        x = self.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)

        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, norm = False, mean = None, std = None):
        super(ResNet, self).__init__()
        self._num_classes = num_classes
        self.in_planes = 64
        self.norm = norm
        self.mean = mean
        self.std = std
        self.block_expansion = block.expansion
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion*25, self._num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    @property
    def num_classes(self):
        return self._num_classes

    @num_classes.setter
    def num_classes(self, value):
        self._num_classes = value
        self.linear = nn.Linear(512 * self.block_expansion, self._num_classes).to(self.linear.weight.device)

    def forward(self, x, is_eval=False):
        if self.norm == True:
            x = Normalization(x, self.mean, self.std)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        if is_eval==False:
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            return out
        else:
            return out.view(out.size(0), -1)

def ResNet18(Num_class=10, Norm=True, norm_mean=None, norm_std=None):
    return ResNet(BasicBlock, [2,2,2,2], num_classes=Num_class, norm=Norm, mean=norm_mean, std=norm_std)

def ResNet34(Num_class=10, Norm=True, norm_mean=None, norm_std=None):
    return ResNet(BasicBlock, [3,4,6,3], num_classes=Num_class, norm=Norm, mean=norm_mean, std=norm_std)

def ResNet50(Num_class=10, Norm=True, norm_mean=None, norm_std=None):
    return ResNet(Bottleneck, [3,4,6,3], num_classes=Num_class, norm=Norm, mean=norm_mean, std=norm_std)

def ResNet101(Num_class=10, Norm=True, norm_mean=None, norm_std=None):
    return ResNet(Bottleneck, [3,4,23,3], num_classes=Num_class, norm=Norm, mean=norm_mean, std=norm_std)

def ResNet152(Num_class=10, Norm=True, norm_mean=None, norm_std=None):
    return ResNet(Bottleneck, [3,8,36,3], num_classes=Num_class, norm=Norm, mean=norm_mean, std=norm_std)


cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):

    def __init__(self, features, num_classes=100, norm = False, mean = None, std = None):
        super().__init__()
        self.features = features
        self.norm = norm
        self.mean = mean
        self.std = std
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        if self.norm == True:
            x = Normalization(x, self.mean, self.std)
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output

def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)

def vgg11_bn(Num_class=10, Norm=True, norm_mean=None, norm_std=None):
    return VGG(make_layers(cfg['A'], batch_norm=True), num_classes=Num_class, norm=Norm, mean=norm_mean, std=norm_std)

def vgg13_bn(Num_class=10, Norm=True, norm_mean=None, norm_std=None):
    return VGG(make_layers(cfg['B'], batch_norm=True), num_classes=Num_class, norm=Norm, mean=norm_mean, std=norm_std)

def vgg16_bn(Num_class=10, Norm=True, norm_mean=None, norm_std=None):
    return VGG(make_layers(cfg['D'], batch_norm=True), num_classes=Num_class, norm=Norm, mean=norm_mean, std=norm_std)

def vgg19_bn(Num_class=10, Norm=True, norm_mean=None, norm_std=None):
    return VGG(make_layers(cfg['E'], batch_norm=True), num_classes=Num_class, norm=Norm, mean=norm_mean, std=norm_std)


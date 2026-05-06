import torch
import torch.nn as nn
import os
import urllib.request
import numpy as np
from PIL import Image

class Bottleneck(nn.Module):
    def __init__(self, inplanes, midplanes, outplanes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, midplanes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(midplanes)
        self.conv2 = nn.Conv2d(midplanes, midplanes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(midplanes)
        self.conv3 = nn.Conv2d(midplanes, outplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample if downsample else nn.Identity()

    def forward(self, x):
        x_skip = self.downsample(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x += x_skip
        x = self.relu(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, inplanes, midplanes):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, midplanes, kernel_size=1)
        self.gn1 = nn.BatchNorm2d(midplanes)
        self.conv2 = nn.Conv2d(midplanes, midplanes, kernel_size=3, padding=1)
        self.gn2 = nn.BatchNorm2d(midplanes)
        self.conv3 = nn.Conv2d(midplanes, inplanes, kernel_size=1)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x_skip = x
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.gn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x += x_skip
        x = self.relu(x)
        return x

def make_layer(inplanes, midplanes, outplanes, blocks, stride):
    downsample = None
    if stride != 1 or inplanes != outplanes:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(outplanes))

    layers = [Bottleneck(inplanes, midplanes, outplanes, stride, downsample)]
    for _ in range(1, blocks):
        layers.append(Bottleneck(outplanes, midplanes, outplanes, stride=1, downsample=None))

    return nn.Sequential(*layers)

class ResNet50(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = make_layer(64, 64, 256, 3, stride=1)
        self.layer2 = make_layer(256, 128, 512, 4, stride=2)
        self.layer3 = make_layer(512, 256, 1024, 6, stride=2)
        self.layer4 = make_layer(1024, 512, 2048, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

class RES50MAT(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = ResNet50()

        self.start_conv0 = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1), nn.PReLU(32))

        self.start_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 2, 1), nn.PReLU(32),
            nn.Conv2d(32, 48, 3, 1, 1), nn.PReLU(48))

        self.start_conv2 = nn.Conv2d(48, 64, 3, 2, 1)

        self.l1 = resnet.layer1
        self.l2 = resnet.layer2
        self.l3 = resnet.layer3
        self.l4 = resnet.layer4

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=1))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=256 + 1024, out_channels=256, kernel_size=1),
            ResBlock(256, 128), ResBlock(256, 128), ResBlock(256, 128))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256 + 512, out_channels=256, kernel_size=1),
            ResBlock(256, 128), ResBlock(256, 128), ResBlock(256, 128))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256 + 256, out_channels=128, kernel_size=1),
            ResBlock(128, 64), ResBlock(128, 64), ResBlock(128, 64))
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=128 + 48, out_channels=64, kernel_size=3, padding=1), nn.PReLU(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), nn.PReLU(64),
            nn.Conv2d(in_channels=64, out_channels=48, kernel_size=3, padding=1), nn.PReLU(48))
        self.convo = nn.Sequential(
            nn.Conv2d(in_channels=48 + 6 + 32, out_channels=32, kernel_size=3, padding=1), nn.PReLU(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1), nn.PReLU(32),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1))
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x, y):
        inputs = torch.cat((x, y), 1)
        x0 = self.start_conv0(inputs)
        x = self.start_conv1(x0)
        x_ = self.start_conv2(x)
        x1 = self.l1(x_)
        x2 = self.l2(x1)
        x3 = self.l3(x2)
        x4 = self.l4(x3)
        X4 = self.conv1(x4)
        X3 = self.up(X4)
        X3 = torch.cat((x3, X3), 1)
        X3 = self.conv2(X3)
        X2 = self.up(X3)
        X2 = torch.cat((x2, X2), 1)
        X2 = self.conv3(X2)
        X1 = self.up(X2)
        X1 = torch.cat((x1, X1), 1)
        X1 = self.conv4(X1)
        X0 = self.up(X1)
        X0 = torch.cat((X0, x), 1)
        X0 = self.conv5(X0)
        X = self.up(X0)
        X = torch.cat((inputs, X, x0), 1)
        alpha = self.convo(X)
        alpha = alpha.clip(0, 1)
        return alpha

    def predict(self, image, trimap):
        assert isinstance(image, Image.Image), f"Expected PIL Image instead of {type(image)}"
        assert isinstance(trimap, Image.Image), f"Expected PIL Image instead of {type(trimap)}"

        image = np.array(image.convert("RGB"))
        trimap = np.array(trimap.convert("L"))

        # Pad image and trimap size to multiples of 64
        h, w, _ = image.shape
        new_h = (((h - 1) // 64) + 2) * 64
        new_w = (((w - 1) // 64) + 2) * 64
        pad_h = new_h - h
        pad_w = new_w - w
        pad_h1 = pad_h // 2
        pad_w1 = pad_w // 2
        pad_h2 = pad_h - pad_h1
        pad_w2 = pad_w - pad_w1
        padding = [(pad_h1, pad_h2), (pad_w1, pad_w2), (0, 0)]

        image = np.pad(image, padding, "symmetric")
        trimap = np.pad(trimap, padding[:2], "symmetric")

        is_bg = trimap == 0
        is_fg = trimap == 255
        is_unknown = ~(is_bg | is_fg)

        trimap = np.array([[is_bg, is_unknown, is_fg]], np.float32)

        with torch.no_grad():
            device = self.start_conv2.weight.device

            image = image.transpose(2, 0, 1)
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).to(device).unsqueeze(0)
            trimap = torch.from_numpy(trimap).to(device)

            alpha = self.forward(image, trimap)

            alpha = alpha.detach().cpu().numpy()[0, 0]

            # Set known foreground and background from trimap
            alpha[is_fg] = 1
            alpha[is_bg] = 0
            
            # Remove padding
            alpha = alpha[pad_h1:pad_h1 + h, pad_w1:pad_w1 + w]

            return Image.fromarray((alpha * 255).astype(np.uint8))

def load(path, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    directory, filename = os.path.split(path)

    if not os.path.exists(path):
        assert filename in ["model.ckpt", "model_better.ckpt"], f"Filename must be model.ckpt or model_better.ckpt instead of {filename}"

        url = f"https://huggingface.co/Coldswamp/ResNet50-Matting/resolve/main/{filename}"

        print(f"Downloading {url}")

        if directory:
            os.path.makedirs(directory, exist_ok=True)

        urllib.request.urlretrieve(url, path)

    d = torch.load(path, weights_only=True, map_location="cpu")["model"]

    model = RES50MAT()
    model.load_state_dict(d)
    model.to(device)
    model.eval()

    return model

def test_shape():
    model = RES50MAT()
    b = torch.randn(1, 3, 1024, 1024)
    c = torch.randn(1, 3, 1024, 1024)

    model.eval()

    with torch.no_grad():
        pred = model(b, c)
        assert pred.shape == (1, 1, 1024, 1024)

if __name__ == "__main__":
    test_shape()

import torch
from torch import Tensor
import torch.nn as nn
import os
import urllib.request
from PIL import Image
import numpy as np


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv1x1s(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample=None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer=None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample=None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer=None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
            self,
            block,
            layers,
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation=None,
            norm_layer=None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
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


class ResBlock(nn.Module):
    def __init__(self, inc, midc, stride=1):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(inc, midc, kernel_size=1, stride=1, padding=0, bias=True)
        self.gn1 = nn.BatchNorm2d(midc)
        self.conv2 = nn.Conv2d(midc, midc, kernel_size=3, stride=1, padding=1, bias=True)
        self.gn2 = nn.BatchNorm2d(midc)
        self.conv3 = nn.Conv2d(midc, inc, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x_ = x
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.gn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = x + x_
        x = self.relu(x)
        return x


def _resnet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


class RES50MAT(nn.Module):
    def __init__(self):
        super(RES50MAT, self).__init__()
        resnet = _resnet50()

        self.start_conv0 = nn.Sequential(nn.Conv2d(6, 32, 3, 1, 1), nn.PReLU(32))

        self.start_conv1 = nn.Sequential(nn.Conv2d(32, 32, 3, 2, 1), nn.PReLU(32), nn.Conv2d(32, 48, 3, 1, 1),
                                         nn.PReLU(48))

        self.start_conv2 = nn.Conv2d(48, 64, 3, 2, 1)

        self.l1 = resnet.layer1
        self.l2 = resnet.layer2
        self.l3 = resnet.layer3
        self.l4 = resnet.layer4

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=1, stride=1, padding=0, bias=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=256 + 1024, out_channels=256, kernel_size=1, stride=1, padding=0, bias=True),
            ResBlock(256, 128), ResBlock(256, 128), ResBlock(256, 128))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256 + 512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=True),
            ResBlock(256, 128), ResBlock(256, 128), ResBlock(256, 128))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256 + 256, out_channels=128, kernel_size=1, stride=1, padding=0, bias=True),
            ResBlock(128, 64), ResBlock(128, 64), ResBlock(128, 64))
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=128 + 48, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.PReLU(64), nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.PReLU(64), nn.Conv2d(in_channels=64, out_channels=48, kernel_size=3, stride=1, padding=1, bias=True),
            nn.PReLU(48))
        self.convo = nn.Sequential(
            nn.Conv2d(in_channels=48 + 6 + 32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.PReLU(32), nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.PReLU(32), nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True))
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

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
        alpha = torch.clamp(alpha, 0, 1)
        return alpha

    def predict(self, image, trimap):
        assert isinstance(image, Image.Image), f"Expected PIL Image instead of {type(image)}"
        assert isinstance(trimap, Image.Image), f"Expected PIL Image instead of {type(trimap)}"

        rawimg = np.array(image.convert("RGB"))
        trimap = np.array(trimap.convert("L"))

        #image = image.permute(2, 0, 1).unsqueeze(0)
        #trimap = trimap.unsqueeze(0).unsqueeze(0)

        rawimg = rawimg[:, :, ::-1]

        import cv2

        trimap_nonp=trimap.copy()
        h,w,c=rawimg.shape
        nonph,nonpw,_=rawimg.shape
        newh= (((h-1)//64)+2)*64
        neww= (((w-1)//64)+2)*64
        padh=newh-h
        padh1=int(padh/2)
        padh2=padh-padh1
        padw=neww-w
        padw1=int(padw/2)
        padw2=padw-padw1
        rawimg_pad=cv2.copyMakeBorder(rawimg,padh1,padh2,padw1,padw2,cv2.BORDER_REFLECT)
        trimap_pad=cv2.copyMakeBorder(trimap,padh1,padh2,padw1,padw2,cv2.BORDER_REFLECT)
        h_pad,w_pad,_=rawimg_pad.shape
        tritemp = np.zeros([*trimap_pad.shape, 3], np.float32)
        tritemp[:, :, 0] = (trimap_pad == 0)
        tritemp[:, :, 1] = (trimap_pad == 128)
        tritemp[:, :, 2] = (trimap_pad == 255)
        tritemp2=np.transpose(tritemp,(2,0,1))
        tritemp2=tritemp2[np.newaxis,:,:,:]
        img=np.transpose(rawimg_pad,(2,0,1))[np.newaxis,::-1,:,:]
        img=np.array(img,np.float32)
        img=img/255.

        with torch.no_grad():
            # TODO device from weights
            device = self.device

            img=torch.from_numpy(img).to(device)
            tritemp2=torch.from_numpy(tritemp2).to(device)

            pred=self.forward(img,tritemp2)

            pred=pred.detach().cpu().numpy()[0]
            pred=pred[:,padh1:padh1+h,padw1:padw1+w]
            preda=pred[0:1,]*255
            preda=np.transpose(preda,(1,2,0))
            preda=preda*(trimap_nonp[:,:,None]==128)+(trimap_nonp[:,:,None]==255)*255

        preda=np.array(preda,np.uint8)[:, :, 0]

        return Image.fromarray(preda)

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

    state_dict = torch.load(path, weights_only=True, map_location="cpu")

    model = RES50MAT()
    model.load_state_dict(state_dict["model"])
    model.to(device)
    model.eval()
    model.device = device

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

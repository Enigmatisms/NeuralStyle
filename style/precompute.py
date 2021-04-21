"""
- @author: Enigmatisms
- @date: 2021.4.19
- Pre-compute step: For 
    - Style img: extracting the style Gram Matrix
    - Content img: extracting content up to conv4_2
---
### VGG 19
==== Conv 1 ====
 2:  Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
 3:  ReLU(inplace=True)
 4:  Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
 5:  ReLU(inplace=True)
 6:  MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
==== Conv 2 ====
 7:  Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
 8:  ReLU(inplace=True)
 9:  Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
10:  ReLU(inplace=True)
11:  MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
==== Conv 3 ====
12:  Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
13:  ReLU(inplace=True)
14:  Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
15:  ReLU(inplace=True)
16:  Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
17:  ReLU(inplace=True)
18:  Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
19:  ReLU(inplace=True)
==== Conv 4 ====
20:  MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
21:  Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
22:  ReLU(inplace=True)
23:  Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
24:  ReLU(inplace=True)
25:  Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
26:  ReLU(inplace=True)
27:  Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
28:  ReLU(inplace=True)
==== Conv 5 ====
29:  MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
30:  Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
31:  ReLU(inplace=True)
32:  Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
33:  ReLU(inplace=True)
34:  Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
35:  ReLU(inplace=True)
36:  Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
37:  ReLU(inplace=True)
"""

import torch
from torch import nn
from torchvision.models import vgg19

class ContentExtractor(nn.Module):
    """
        Extracting content representation using pretrained vgg-16 upto conv4_2
        ---
        ### Param
        - img: Content img `torch.Tensor`
    """
    def __init__(self):
        super().__init__()
        model = vgg19(pretrained = True)
        layers = [layer for layer in model.modules()]
        self.convs = nn.Sequential(
            *(layers[i] for i in range(2, 29))
        )
        # pre-trained network requires no grad
        for p in self.convs.parameters():
            p.requires_grad = False
    
    def forward(self, x):
        if x.dim() == 3:
            x = x[None, :, :, :]
        return self.convs(x)


class StyleExtractor(nn.Module):
    """
        Extracting style representation using pretrained vgg-16 using forward
        ---
        ### Param
        - img: Content img `torch.Tensor`
    """
    def __init__(self):
        super().__init__()
        model = vgg19(pretrained = True)
        for p in model.parameters():    # parameter in the Net itself requires no grad 
            p.requires_grad = False
        layers = [layer for layer in model.modules()]
        self.conv1 = nn.Sequential(
            *(layers[i] for i in range(2, 6))
        )       # output before MaxPool2d

        self.conv2 = nn.Sequential(
            *(layers[i] for i in range(6, 11))
        )

        self.conv3 = nn.Sequential(
            *(layers[i] for i in range(11, 20))
        )

        self.conv4 = nn.Sequential(
            *(layers[i] for i in range(20, 29))
        )

        self.conv5 = nn.Sequential(
            *(layers[i] for i in range(29, 38))
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x[None, :, :, :]
        matrices = []
        x = self.conv1(x)
        gram = StyleExtractor.computeGramMatrix(x)
        matrices.append(gram)
        x = self.conv2(x)
        gram = StyleExtractor.computeGramMatrix(x)
        matrices.append(gram)
        x = self.conv3(x)
        gram = StyleExtractor.computeGramMatrix(x)
        matrices.append(gram)
        x = self.conv4(x)
        gram = StyleExtractor.computeGramMatrix(x)
        matrices.append(gram)
        x = self.conv5(x)
        gram = StyleExtractor.computeGramMatrix(x)
        matrices.append(gram)
        return matrices

    """
        ### Return
        - N: Number of filters (channels) for this feature map
        - M: Size (width times height) of this feature map
        - x @ x.T Gram Matrix itself
    """
    @staticmethod
    def computeGramMatrix(x):
        flat = x.view(x.shape[1], -1).clone()
        N, M = flat.shape
        return (N, M, flat @ flat.T)


        




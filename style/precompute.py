"""
- @author: Enigmatisms
- @date: 2021.4.19
- Pre-compute step: For 
    - Style img: extracting the style Gram Matrix
    - Content img: extracting content up to conv4_2
- It seems that the original paper depends on pre-trained VGG-19
---
### VGG 16
    Conv1
2         <class 'torch.nn.modules.conv.Conv2d'>
3         <class 'torch.nn.modules.activation.ReLU'>
4         <class 'torch.nn.modules.conv.Conv2d'>
5         <class 'torch.nn.modules.activation.ReLU'>
6         <class 'torch.nn.modules.pooling.MaxPool2d'>
    Conv2
7         <class 'torch.nn.modules.conv.Conv2d'>
8         <class 'torch.nn.modules.activation.ReLU'>
9         <class 'torch.nn.modules.conv.Conv2d'>
10         <class 'torch.nn.modules.activation.ReLU'>
11        <class 'torch.nn.modules.pooling.MaxPool2d'>
    Conv3
12        <class 'torch.nn.modules.conv.Conv2d'>
13        <class 'torch.nn.modules.activation.ReLU'>
14        <class 'torch.nn.modules.conv.Conv2d'>
15        <class 'torch.nn.modules.activation.ReLU'>
16        <class 'torch.nn.modules.conv.Conv2d'>
17        <class 'torch.nn.modules.activation.ReLU'>
18        <class 'torch.nn.modules.pooling.MaxPool2d'>
    Conv4
19        <class 'torch.nn.modules.conv.Conv2d'>
20        <class 'torch.nn.modules.activation.ReLU'>
21        <class 'torch.nn.modules.conv.Conv2d'>
22        <class 'torch.nn.modules.activation.ReLU'>
        === Conv 4_2 seperation ===
23        <class 'torch.nn.modules.conv.Conv2d'>
24        <class 'torch.nn.modules.activation.ReLU'>
25        <class 'torch.nn.modules.pooling.MaxPool2d'>
    Conv5
26        <class 'torch.nn.modules.conv.Conv2d'>
27        <class 'torch.nn.modules.activation.ReLU'>
28        <class 'torch.nn.modules.conv.Conv2d'>
29        <class 'torch.nn.modules.activation.ReLU'>
30        <class 'torch.nn.modules.conv.Conv2d'>
31        <class 'torch.nn.modules.activation.ReLU'>    Conv5_3
"""

import torch
from torch import nn
from torchvision.models import vgg16

"""
    Extracting content representation using pretrained vgg-16 upto conv4_2
    ---
    ### Param
    - img: Content img `torch.Tensor`
"""
class ContentExtractor(nn.Module):
    def __init__(self, img:torch.Tensor):
        super().__init__()
        model = vgg16(pretrained = True)
        layers = [layer for layer in model.modules()]
        self.convs = nn.Sequential(
            *(layers[i] for i in range(2, 23))
        )
        # pre-trained network requires no grad
        for p in self.convs.parameters():
            p.requires_grad = False
    
    def forward(self, x):
        return self.convs(x)

class StyleExtractor(nn.Module):
    def __init__(self, img):
        super().__init__()
        model = vgg16(pretrained = True)
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
            *(layers[i] for i in range(11, 18))
        )

        self.conv4 = nn.Sequential(
            *(layers[i] for i in range(18, 25))
        )

        self.conv5 = nn.Sequential(
            *(layers[i] for i in range(25, 32))
        )

    def forward(self, x):
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
        flat = x.view(x.shape[1], -1)
        N, M = flat.shape
        return (N, M, x @ x.T)


        




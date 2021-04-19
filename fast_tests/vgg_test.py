import torch
import matplotlib.pyplot as plt
from torch import nn
from torchvision import transforms
from torchvision.models import vgg16

if __name__ == "__main__":
    torch.set_default_tensor_type(torch.FloatTensor)
    tf = transforms.ToTensor()
    model_16 = vgg16(pretrained = True)
    print("VGG 16:")
    for layer in model_16.modules():
        print(layer)
    
    print("Extracted:")
    layers = [layer for layer in model_16.modules()]
    for i, l in enumerate(layers):
        print("%d -> "%(i), type(l))
    print("Layer 2:", layers[2])
    print("Layer 3:", layers[3])
    conv = nn.Sequential(
        *(layers[i] for i in range(2, 4))
    )
    img = plt.imread('../asset/star.jpg').copy()
    tensor = tf(img)
    tensor = tensor[None, :, :, :]
    out = conv(tensor)
    print("Process completed.")
    print(out.shape)
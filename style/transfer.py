#-*-coding:utf-8-*-
"""
    Style Transfer
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import shutil
from datetime import datetime
from torch.autograd import Variable as Var
from torchvision.utils import save_image

from torch import optim
from torch import nn
from torchvision import transforms
from torchvision.transforms.functional import resize
from torch.utils.tensorboard import SummaryWriter

from precompute import *
from lossTerm import *

if __name__ == "__main__":
    alpha = 1e-5
    epoches = 200
    save_time = 20
    del_dir = True
    use_gray_scale_initialization = False

    tf = transforms.ToTensor()

    img_raw = plt.imread("../content/content.jpg")
    img = tf(img_raw.copy())

    style = plt.imread("../asset/star.jpg")
    style = tf(style.copy())
    style = resize(style, img.shape[1:])        # resize style img to the same shape as content img

    sext = StyleExtractor()
    cext = ContentExtractor()
    
    if use_gray_scale_initialization:           # using gray scale img to intialize transfer estimation
        mean = torch.mean(img, dim = 0)[None, :, :]
        gen = Var(torch.vstack(mean, mean, mean))
    else:
        gen = Var(torch.zeros_like(img), requires_grad = True)
    
    logdir = '../logs/'
    if os.path.exists(logdir) and del_dir:
        shutil.rmtree(logdir)
    
    time_stamp = "{0:%Y-%m-%d/%H-%M-%S}-epoch{1}/".format(datetime.now(), epoches)
    writer = SummaryWriter(log_dir = logdir+time_stamp)

    style_gram = sext(style)
    content = cext(img)
    optimizer = optim.Adam([gen, ], lr = 1e-2)
    style_loss_f = StyleLoss()
    content_loss_f = ContentLoss()
    for epoch in range(epoches):
        gen_c = cext(gen)
        gen_s = sext(gen)
        style_loss = style_loss_f(gen_s, style_gram)
        content_loss = content_loss_f(gen_c, content)
        loss = alpha * content_loss + (1 - alpha) * style_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        writer.add_scalar('Loss/Total Loss', loss, epoch)
        writer.add_scalar('Loss/Content loss', content_loss, epoch)
        writer.add_scalar('Loss/Style loss', style_loss, epoch)
        print("Training epoch: %3d / %d \tloss: %.6f"%(epoch, epoches, loss))
        if (epoch + 1) % save_time == 0:
            save_image(gen.detach(), "..\\imgs\\G_%d.jpg"%(epoch), 1)
    writer.close()
    print("Output completed.")


        


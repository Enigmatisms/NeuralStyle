#-*-coding:utf-8-*-
"""
    Style Transfer
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import shutil
import argparse
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type = float, default = 5e-6, help = "Ratio of content loss in the total loss")
    parser.add_argument("--epoches", type = int, default = 400, help = "Training lasts for . epoches (for LBFGS)")
    parser.add_argument("--max_iter", type = int, default = 10, help = "LBFGS max iteration number")
    parser.add_argument("--save_time", type = int, default = 20, help = "Save image every <save_time> epoches")
    parser.add_argument("-d", "--del_dir", action = "store_true", help = "Delete dir ./logs and start new tensorboard records")
    parser.add_argument("-g", "--gray", default = False, action = "store_true", help = "Using grayscale image as initialization for generated image")
    parser.add_argument("-c", "--cuda", default = False, action = "store_true", help = "Use CUDA to speed up training")
    args = parser.parse_args()

    alpha = args.alpha
    epoches = args.epoches
    save_time = args.save_time
    del_dir = args.del_dir
    use_cuda = args.cuda
    use_gray_scale_initialization = args.gray
    max_iter = args.max_iter

    tf = transforms.ToTensor()

    img_raw = plt.imread("../content/content.jpg")
    img = tf(img_raw.copy())

    style = plt.imread("../asset/star.jpg")
    style = tf(style.copy())
    style = resize(style, img.shape[1:])        # resize style img to the same shape as content img
    sext = StyleExtractor()
    cext = ContentExtractor()
    style_loss_f = StyleLoss()
    content_loss_f = ContentLoss()
       
    mean = None
    gen = None
    if use_gray_scale_initialization:           # using gray scale img to intialize transfer estimation
        gray_scale = torch.mean(img, dim = 0)[None, :, :]
        mean = torch.vstack((gray_scale, gray_scale, gray_scale))
    else:
        mean = torch.zeros_like(img)
    if use_cuda and torch.cuda.is_available():
            sext = sext.cuda()
            cext = cext.cuda()
            img = img.cuda()
            style = style.cuda()
            style_loss_f = style_loss_f.cuda()
            content_loss_f = content_loss_f.cuda()
            gen = Var(mean.cuda(), requires_grad = True).cuda()
    else:
        gen = Var(mean, requires_grad = True)
        use_cuda = False
        print("CUDA not available.")
    
    logdir = '../logs/'
    if os.path.exists(logdir) and del_dir:
        shutil.rmtree(logdir)
    
    time_stamp = "{0:%Y-%m-%d/%H-%M-%S}-epoch{1}/".format(datetime.now(), epoches)
    writer = SummaryWriter(log_dir = logdir+time_stamp)

    style_gram = sext(style)
    content = cext(img)
    optimizer = optim.Adam([gen, ], lr = 5e-2)
    for epoch in range(epoches):
        gen_c = cext(gen)
        gen_s = sext(gen)
        style_loss = style_loss_f(gen_s, style_gram)
        content_loss = content_loss_f(gen_c, content)
        loss = alpha * content_loss + (1 - alpha) * style_loss
        writer.add_scalar('Loss/Total Loss', loss, epoch)
        writer.add_scalar('Loss/Content loss', content_loss, epoch)
        writer.add_scalar('Loss/Style loss', style_loss, epoch)
        print("Training epoch: %3d / %d \tloss: %.6f"%(epoch, epoches, loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch % save_time) == 0:
            save_image(gen.detach().clamp_(0, 1), "..\\imgs\\G_%d.jpg"%(epoch + 1), 1)
    writer.close()
    print("Output completed.")
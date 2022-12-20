# NeuralStyle
---
​		My own implementation of CVPR 2016 paper: Image Style Transfer Using Convolutional Neural Networks. This work is, I think, simple but elegant (I mean the paper, not my implementation) with good interpretability. 

- CVPR 2016 OpenAccess Link is here: [CVPR 2016 open access](https://openaccess.thecvf.com/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html)

---

### To run the code

​		Make sure to have Pytorch / Tensorboard on your device, CUDA is available too yet I failed to use it (GPU memory not enough, yet API is good to go). I am currently using Pytorch 1.7.0 + CU101.

​		On Init, it might require you to download pretrained VGG-19 network, which requires network connection. 

---

### Tree - Working Directory 

- folder `content`: Where I keep content images.
- folder `imgs`: To which the output goes.
- folder `style`:
  - `lossTerm.py`: Style loss and Content loss are implemented here.
  - `precompute.py`: VGG-19 utilization, style and content extractors.
  - **`transfer.py`**: executable script.

---

### A Little Help

​		Always run `transfer.py` in folder `style/`, using `python ./transfer.py -h`， you'll get:

```shell
usage: transfer.py [-h] [--alpha ALPHA] [--epoches EPOCHES]
                   [--max_iter MAX_ITER] [--save_time SAVE_TIME] [-d] [-g]
                   [-c]
optional arguments:
  -h, --help            show this help message and exit
  --alpha ALPHA         Ratio of content loss in the total loss
  --epoches EPOCHES     Training lasts for . epoches (for LBFGS)
  --max_iter MAX_ITER   LBFGS max iteration number
  --save_time SAVE_TIME
                        Save image every <save_time> epoches
  -d, --del_dir         Delete dir ./logs and start new tensorboard records
  -g, --gray            Using grayscale image as initialization for generated
                        image
  -c, --cuda            Use CUDA to speed up training
```

---

### Requirements

- Run:

```
python3 -m pip install -r requirements.py
```

​		To find out.

---

### Training Process

- Something strange happened. Loss exploded twice (but recovered.). Tensorboard graphs:

![](imgs/training.JPG)

Therefore, parameter images change like this (Initialized with grayscale image).

| ![](imgs/G_star_71.jpg)  |  ![](imgs/G_star_221.jpg)   |  ![](imgs/G_star_481.jpg)   |
| :----------------------: | :-------------------------: | :-------------------------: |
| ![](imgs/G_chaos_11.jpg) | ![](imgs/G_chaos_181.jpg) | ![](imgs/G_chaos_241.jpg) |
|     First few epochs     | Exploded, for 2th row image |          Recovered          |

---

### Results

- CPU training is tooooooo slow. Took me **<u>2+ hours</u>** for 800 iterations. (i5-8250U 8th Gen @ 1.60Hz)

| <img src="./asset/star.jpg" style="zoom:80%;" /> | ![](./content/content.jpg) | ![](./imgs/G_star_801.jpg) |
| :----------------------------------------------: | :------------------------: | :------------------------: |
| <img src="asset/chaos.jpg" style="zoom:80%;" />  |  ![](content/content.jpg)  | ![](imgs/G_chaos_801.jpg)  |
|                      Style                       |          Content           |   Output(800 Iterations)   |

- I've also done the style transfer of Van Gogh's self portrait for my dad, which is not appropriate to display, but worked.

---

### Possible TODOs

- [ ] Try adding InstanceNorm into VGG-19 ? Useful ? Meaningful ?

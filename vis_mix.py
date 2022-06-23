import numpy as np
import torch
from PIL import Image

x = torch.load('/home/dzy/RepDistiller_V3_MIXUP/pre_mix/mix_up_data_240.pt')
index = np.loadtxt('/home/dzy/RepDistiller_V3_MIXUP/pre_mix/epoch_mix_remain_240.txt')

for i in range(100):
    ii = int(index[i])
    im = Image.fromarray(x[ii].numpy())
    im.save("./mix_up/"+str(i)+".png")
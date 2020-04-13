import torch
import torch.nn.utils.prune as prune
from model import Generator, Discriminator
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np


ngpu = 0
device = torch.device("cuda:0" if(torch.cuda.is_available() and ngpu > 0) else "cpu")
netG = Generator(ngpu)
noise = torch.randn(64, 100, 1, 1, device=device)
img_list = []

with torch.no_grad():
    fake = netG(noise).detach().cpu()
img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

#%%capture
fig = plt.figure(figsize=(8,8))
plt.axis("off")
for i in img_list:
    plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)
plt.show()

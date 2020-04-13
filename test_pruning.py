import torch
import torch.nn.utils.prune as prune
from model import Generator, Discriminator
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

ngpu = 0
device = torch.device("cuda:0" if(torch.cuda.is_available() and ngpu > 0) else "cpu")
netG = Generator(ngpu).to(device)
weights = "./weights/netG.pth"
netG.load_state_dict(torch.load(weights, map_location=device))
noise = torch.randn(64, 100, 1, 1, device=device)

for name, module in netG.named_modules():
    if isinstance(module, torch.nn.ConvTranspose2d):
        prune.l1_unstructured(module, name='weight', amount=0.1)  # 关键就是这句  amount是0-1就是按比例剪枝，
        # amount是int那就按个数剪枝,这里是4x4卷积 乘上 in_channel和out_channel 剪枝个数也就是 in_channel和out_channel的积
        # 但他是按所有卷积的从小到大的顺序排序 不是按选取每个卷积的最小值

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

import cv2
import functools
import json
from math import log10
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import *
from dataset import TestUrbanClassDataset
from utils import *


gen = UNet(input_dim, real_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
disc = PatchDiscriminator(input_dim + real_dim).to(device)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

loaded_state = torch.load(model_path)
gen.load_state_dict(loaded_state["gen"])
gen_opt.load_state_dict(loaded_state["gen_opt"])

test_dataset = TestUrbanClassDataset(img_path, 'a2b')
test_dataloader = DataLoader(test_dataset, batch_size=1,
                        shuffle=False, num_workers=0)

for i, batch in enumerate(test_dataloader):
	with torch.no_grad():
		test_image = batch[0].to(device)
		output = gen( test_image )
		save_image(output, '{}/{}'.format(img_save, batch[2][0][:-4]))

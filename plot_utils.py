import matplotlib.pyplot as plt
import torch
from torchvision.transforms import transforms
import torchvision
from PIL import Image
import time
import os
# ADD IM CONVERT USING PLOT_IMGS CODE, THEN SEPERATE CODE FOR PLOTTING WITH SUBPLOTS (*AGRS)

def plot_imgs(img, seq_no, lp_path, log_process):
    '''
    Display image
    '''
    if torch.is_tensor(img):
        img = img.cpu().clone().detach()
        img = torch.squeeze(img)
        means = torch.reshape(torch.tensor([0.485, 0.456, 0.406]), (3, 1, 1))
        stds =  torch.reshape(torch.tensor([0.229, 0.224, 0.225]), (3, 1, 1))
        img = img * stds + means
        img = torch.clip(img, 0, 1)
        img = transforms.ToPILImage()(img)

    if log_process:
        seq_path = lp_path + '/sequence'
        if not os.path.isdir(seq_path):
            os.makedirs(seq_path)
        img.save("{folder}\\seq_{num}.jpg".format(folder=seq_path, num=seq_no))


    plt.imshow(img)
    plt.show(block=False)
    plt.pause(3) 
    plt.close("all")

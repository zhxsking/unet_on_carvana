# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 16:35:39 2018
将图片中心裁剪为方形
@author: zhxsking
"""

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from os import listdir
from os.path import join
from PIL import Image

class CarvanaDataset(Dataset):
    def __init__(self, dir_img, dir_mask, img_dim=3, scale=0.5):
        super().__init__()
        self.dir_img = dir_img
        self.dir_mask = dir_mask
        self.img_dim = img_dim
        self.scale = scale
        
    def __getitem__(self, index):
        img_names = listdir(self.dir_img)
        mask_names = listdir(self.dir_mask)
        
        read_mode = 'L' if self.img_dim==1 else ('RGB' if self.img_dim==3 else 'error')
        img = Image.open(join(self.dir_img, img_names[index])).convert(read_mode)
        mask = Image.open(join(self.dir_mask, mask_names[index])).convert('L')
        
#        img = img.resize(tuple(map(lambda x: int(x * self.scale), img.size)))
#        mask = mask.resize(tuple(map(lambda x: int(x * self.scale), img.size)))
        
        new_size = tuple(map(lambda x: int(x * self.scale), (img.height, img.width)))
        crop_size = min(new_size)
        
        process = transforms.Compose([
                transforms.Resize(new_size),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                ])
        
        img = process(img)
        mask = process(mask)
        
        return img, mask
    
    def __len__(self):
        return len(listdir(self.dir_img))
    
if __name__ == '__main__':
    __spec__ = None
    
    import torch
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    torch.manual_seed(1)
    
    dir_img = r"E:\pic\carvana\just_for_test\train"
    dir_mask = r"E:\pic\carvana\just_for_test\train_masks"
    dataset = CarvanaDataset(dir_img, dir_mask)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=2)
    
    dataset_iter = iter(dataset)
    img_o, lab_o = dataset_iter.__next__()
    img = img_o.numpy()
    lab = lab_o.numpy()
    plt.figure(1)
    plt.subplot(221)
    plt.imshow(img[0], cmap='gray')
    plt.subplot(222)
    plt.imshow(lab[0], cmap='gray')
    
    dataloader_iter = iter(dataloader)
    img1_o, lab1_o = dataloader_iter.__next__()
    img1 = img1_o.numpy()
    lab1 = lab1_o.numpy()
    plt.subplot(223)
    plt.imshow(img1[0][0], cmap='gray')
    plt.subplot(224)
    plt.imshow(lab1[0][0], cmap='gray')
    

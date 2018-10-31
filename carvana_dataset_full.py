# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 16:35:39 2018
将图片裁剪成左右两个方形处理
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
        
        img = img.resize(tuple(map(lambda x: int(x * self.scale), img.size)))
        mask = mask.resize(tuple(map(lambda x: int(x * self.scale), mask.size)))
        
        # 图片的左半边，640*640，参数为左上右下
        img_left = img.crop((0, 0, img.height, img.height))
        mask_left = mask.crop((0, 0, mask.height, mask.height))
        # 图片的右半边，640*640
        img_right = img.crop((img.width-img.height, 0, img.width, img.height))
        mask_right = mask.crop((mask.width-mask.height, 0, mask.width, mask.height))
        
        new_size = tuple(map(lambda x: int(x * self.scale), (img_left.height, img_left.width)))
        
        process = transforms.Compose([
                transforms.Resize(new_size),
                transforms.ToTensor(),
                ])
        
        img_left = process(img_left)
        mask_left = process(mask_left)
        img_right = process(img_right)
        mask_right = process(mask_right)
        
        return img_left, mask_left, img_right, mask_right
    
    def __len__(self):
        return len(listdir(self.dir_img))
    
if __name__ == '__main__':
    __spec__ = None
    
    import torch
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    torch.manual_seed(1)
    
    dir_img = r"D:\pic\carvana\just_for_test\train"
    dir_mask = r"D:\pic\carvana\just_for_test\train_masks"
    dataset = CarvanaDataset(dir_img, dir_mask)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=2)
    
    dataset_iter = iter(dataset)
    img_l, lab_l, img_r, lab_r = dataset_iter.__next__()
    img_le = img_l.numpy()
    lab_le = lab_l.numpy()
    img_re = img_r.numpy()
    lab_re = lab_r.numpy()
    plt.figure(1)
    plt.subplot(241)
    plt.imshow(img_le[0], cmap='gray')
    plt.subplot(242)
    plt.imshow(lab_le[0], cmap='gray')
    plt.subplot(243)
    plt.imshow(img_re[0], cmap='gray')
    plt.subplot(244)
    plt.imshow(lab_re[0], cmap='gray')
    
    dataloader_iter = iter(dataloader)
    img1_l, lab1_l, img1_r, lab1_r = dataloader_iter.__next__()
    img1_le = img1_l.numpy()
    lab1_le = lab1_l.numpy()
    img1_re = img1_r.numpy()
    lab1_re = lab1_r.numpy()
    plt.subplot(245)
    plt.imshow(img1_le[0][0], cmap='gray')
    plt.subplot(246)
    plt.imshow(lab1_le[0][0], cmap='gray')
    plt.subplot(247)
    plt.imshow(img1_re[0][0], cmap='gray')
    plt.subplot(248)
    plt.imshow(lab1_re[0][0], cmap='gray')
    

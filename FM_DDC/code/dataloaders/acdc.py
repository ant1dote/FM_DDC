from dataloaders.transform import random_rot_flip, random_rotate, blur, obtain_cutmix_box, random_rot_flip_rgb

from copy import deepcopy
import h5py
import math
import numpy as np
import os
from PIL import Image, ImageFilter
import random
from scipy.ndimage.interpolation import zoom
from scipy import ndimage
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms
import pywt
from math import sqrt

def fundus_blur(img, p=1.0):
    if random.random() < p:
        sigma = np.random.uniform(1.0, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img

def hflip(img, mask, p=0.5):
    if random.random() < p:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return img, mask

def vflip(img, mask, p=0.5):
    if random.random() < p:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
    return img, mask

def rotate(img, mask, p=0.5):
    if random.random() < p:
        img = img.transpose(Image.ROTATE_90)
        mask = mask.transpose(Image.ROTATE_90)
    return img, mask



def random_rot_flip(img, mask):
    k = np.random.randint(0, 4)
    img = np.rot90(img, k)
    mask = np.rot90(mask, k)
    axis = np.random.randint(0, 2)
    img = np.flip(img, axis=axis).copy()
    mask = np.flip(mask, axis=axis).copy()
    return img, mask

def random_rotate(img, mask):
    angle = np.random.randint(-20, 20)
    img = ndimage.rotate(img, angle, order=0, reshape=False)
    mask = ndimage.rotate(mask, angle, order=0, reshape=False)
    return img, mask



class Skin_DIFF_FSP(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size

        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            with open(self.root+'/val_set_new_3.txt', 'r') as f:
                self.ids = f.read().splitlines()

    def freq_comp_perturb(self, freq_comp):
        hf_min, hf_max = np.min(freq_comp), np.max(freq_comp)
        freq_comp = (((freq_comp - hf_min) / (hf_max - hf_min)) * 255).astype(np.uint8)
        freq_comp = Image.fromarray(freq_comp.transpose((1,2,0)))           
        freq_comp = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(freq_comp)       
        freq_comp = np.array(freq_comp).astype(np.float32)
        min_blur_hf1, max_blur_hf1 = np.min(freq_comp), np.max(freq_comp)
        freq_comp = hf_min + ((freq_comp - min_blur_hf1) / (max_blur_hf1 - min_blur_hf1)) * (hf_max - hf_min)
        freq_comp = freq_comp.transpose((2, 0, 1))
        return freq_comp

    def __getitem__(self, item):
        
        id = self.ids[item].split('.')[0]+'.h5'
        
        if 'train' in self.mode:
            h5f = h5py.File(self.root + "/train_new_3/{}".format(id), 'r')
        if self.mode == 'val':
            h5f = h5py.File(self.root + "/val_new_3/{}".format(id), 'r')
            img, mask = h5f['image'][:], h5f['label'][:]
            return img, mask

        img, mask = h5f['image'][:], h5f['label'][:]
        img = Image.fromarray((img.transpose((1, 2, 0)) * 255).astype(np.uint8))
        mask = Image.fromarray(mask * 255)
        img, label = img.resize((224, 224), Image.BICUBIC), label.resize((224, 224), Image.NEAREST) 

        if random.random() > 0.5:
            img, mask = hflip(img, mask)
        if random.random() > 0.5:
            img, mask = vflip(img, mask)
        if random.random() > 0.5:
            img, mask = rotate(img, mask)
       
        if self.mode == 'train_l':
            img, mask = np.asarray(img) / 255.0, np.asarray(mask) // 255
            img = img.transpose((2, 0, 1))
            return torch.from_numpy(img).float(), torch.from_numpy(np.array(mask)).long()
        
        img_s1, img_s2 = deepcopy(img), deepcopy(img)
                    
        img_s1 = blur(img_s1, p=0.5)

        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)

        cutmix_box1 = obtain_cutmix_box(self.size, p=0.5)
        cutmix_box2 = obtain_cutmix_box(self.size, p=0.5)
        
        img_s2 = np.asarray(img_s2).transpose((2, 0, 1)).astype(np.float32) / 255.0
        
        coeffs_s2 = pywt.dwt2(img_s2, 'haar')
        ll_s2, details_s2 = coeffs_s2
        
        ll_s2 = self.freq_comp_perturb(ll_s2)    
            
        list_details_s2 = list(details_s2)
            
        list_details_s2[0] = self.freq_comp_perturb(list_details_s2[0])  
        list_details_s2[1] = self.freq_comp_perturb(list_details_s2[1])        
        list_details_s2[2] = self.freq_comp_perturb(list_details_s2[2])
           
        details_s2 = tuple(list_details_s2)
        img_s2 = pywt.idwt2((ll_s2, details_s2), wavelet='haar')
            
        img_min, img_max = np.min(img_s2), np.max(img_s2)
        img_s2_png = (((img_s2 - img_min) / (img_max - img_min)) * 255).astype(np.uint8)
        img_s2_png = Image.fromarray(img_s2_png.transpose((1, 2, 0)).astype(np.uint8), mode='RGB')
    
        img_s1_copy = deepcopy(img_s1)
        img_s1_copy = np.asarray(img_s1_copy)

        img_s2_png_copy = np.asarray(img_s2_png)
            
        img_diff_2 = abs(img_s2_png_copy - img_s1_copy)
        
        img_add_2 = img_s2_png_copy + img_diff_2 
        img_add_2 = img_add_2.transpose((2, 0, 1)) / 255.0
        
        img = torch.from_numpy(np.array(img)).permute((2, 0, 1)).float() / 255.0
        img_s1 = torch.from_numpy(np.array(img_s1)).permute((2, 0, 1)).float() / 255.0
        img_add_2 = torch.from_numpy(img_add_2).float()
        return img, img_s1, img_add_2, cutmix_box1, cutmix_box2
        
    def __len__(self):
        return len(self.ids)
    



class ACDC_DIFF_FSP(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size

        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            with open(self.root+'/val.list', 'r') as f:
                self.ids = f.read().splitlines()

    def freq_comp_perturb(self, freq_comp):
        hf_min, hf_max = np.min(freq_comp), np.max(freq_comp)
        freq_comp = (((freq_comp - hf_min) / (hf_max - hf_min)) * 255).astype(np.uint8)
        freq_comp = Image.fromarray(freq_comp)           
        freq_comp = transforms.ColorJitter(0.0, (0.5, 1.), 0, 0.)(freq_comp)      
        freq_comp = blur(freq_comp) 
        freq_comp = np.array(freq_comp).astype(np.float32)
        min_blur_hf1, max_blur_hf1 = np.min(freq_comp), np.max(freq_comp)
        freq_comp = hf_min + ((freq_comp - min_blur_hf1) / (max_blur_hf1 - min_blur_hf1)) * (hf_max - hf_min)
        
        return freq_comp

    def __getitem__(self, item):
        
        id = self.ids[item]
        if self.mode !='val':
            sample = h5py.File(self.root+id, 'r')
        else:
            sample = h5py.File(os.path.join(self.root+'/data/', id+'.h5'), 'r')
        img = sample['image'][:]
        mask = sample['label'][:]

        if self.mode == 'val':
            return torch.from_numpy(img).float(), torch.from_numpy(mask).long()

        if random.random() > 0.5:
            img, mask = random_rot_flip(img, mask)
        elif random.random() > 0.5:
            img, mask = random_rotate(img, mask)
        
        x, y = img.shape
        img = zoom(img, (self.size / x, self.size / y), order=0)
        mask = zoom(mask, (self.size / x, self.size / y), order=0)

        if self.mode == 'train_l':
            return torch.from_numpy(img).unsqueeze(0).float(), torch.from_numpy(np.array(mask)).long()

        img = Image.fromarray((img * 255).astype(np.uint8))
        
        img_s1, img_s2 = deepcopy(img), deepcopy(img)
        img_s1_ =  deepcopy(img)
        img_s1_ = transforms.ColorJitter(0.0, (0.5, 1.0), 0.0, 0.0)(img_s1_)
        img = torch.from_numpy(np.array(img)).float() / 255.0
    
        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(self.size, p=0.5)
        
        img_s1_copy_ = deepcopy(img_s1_)
        img_s1_copy_ = np.asarray(img_s1_copy_)
        
        img_s1 = torch.from_numpy(np.array(img_s1)).float() / 255.0

        cutmix_box2 = obtain_cutmix_box(self.size, p=0.5)
            
        img_s2 = np.asarray(img_s2).astype(np.float32) / 255.0
        coeffs_s2 = pywt.dwt2(img_s2, 'coif2')
        ll_s2, details_s2 = coeffs_s2
        
        ll_s2 = self.freq_comp_perturb(ll_s2)    
            
        list_details_s2 = list(details_s2)
            
        list_details_s2[0] = self.freq_comp_perturb(list_details_s2[0])  
        list_details_s2[1] = self.freq_comp_perturb(list_details_s2[1])        
        list_details_s2[2] = self.freq_comp_perturb(list_details_s2[2])
           
        details_s2 = tuple(list_details_s2)
        img_s2 = pywt.idwt2((ll_s2, details_s2), wavelet='coif2')
            
        img_min, img_max = np.min(img_s2), np.max(img_s2)
        img_s2_png = (((img_s2 - img_min) / (img_max - img_min)) * 255).astype(np.uint8)
        img_s2_png = Image.fromarray(img_s2_png.astype(np.uint8), mode='L')
            
        img_s2_png_copy = np.asarray(img_s2_png)
            
        img_diff_2 = abs(img_s2_png_copy - img_s1_copy_)
        
        img_add_2 = img_s2_png_copy + img_diff_2 #* 1.2
            
        img_add_2 = torch.from_numpy(img_add_2).float() / 255

        return img.unsqueeze(0), img_s1.unsqueeze(0), img_add_2.unsqueeze(0), cutmix_box1, cutmix_box2
        
        
    def __len__(self):
        return len(self.ids)


class Promise_FSP(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size

        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            with open(self.root+'/val_3.txt', 'r') as f:
                self.ids = f.read().splitlines()

    def freq_comp_perturb(self, freq_comp):
        hf_min, hf_max = np.min(freq_comp), np.max(freq_comp)
        freq_comp = (((freq_comp - hf_min) / (hf_max - hf_min)) * 255).astype(np.uint8)
        freq_comp = Image.fromarray(freq_comp)           
        freq_comp = transforms.ColorJitter(0.0, (0.2, 0.5), 0, 0.)(freq_comp)      
        freq_comp = blur(freq_comp) 
        freq_comp = np.array(freq_comp).astype(np.float32)
        min_blur_hf1, max_blur_hf1 = np.min(freq_comp), np.max(freq_comp)
        freq_comp = hf_min + ((freq_comp - min_blur_hf1) / (max_blur_hf1 - min_blur_hf1)) * (hf_max - hf_min)
        
        return freq_comp

    def __getitem__(self, item):
        
        id = self.ids[item]
        if self.mode !='val':
            sample = h5py.File(self.root+'/train_3/'+id+'.h5', 'r')
        else:
            sample = h5py.File(os.path.join(self.root+'/val_3/', id+'.h5'), 'r')
        img = sample['image'][:]
        mask = sample['label'][:]

        if self.mode == 'val':
            return torch.from_numpy(img).float(), torch.from_numpy(mask).long()

        if random.random() > 0.5:
            img, mask = random_rot_flip(img, mask)
        elif random.random() > 0.5:
            img, mask = random_rotate(img, mask)
        
        x, y = img.shape
        img = zoom(img, (self.size / x, self.size / y), order=0)
        mask = zoom(mask, (self.size / x, self.size / y), order=0)

        if self.mode == 'train_l':
            return torch.from_numpy(img).unsqueeze(0).float(), torch.from_numpy(np.array(mask)).long()

        img = Image.fromarray((img * 255).astype(np.uint8))
        
        img_s1, img_s2 = deepcopy(img), deepcopy(img)
        img_s1_ =  deepcopy(img)
        img_s1_ = transforms.ColorJitter(0.0, (0.2, 0.5), 0.0, 0.0)(img_s1_)
        img = torch.from_numpy(np.array(img)).float() / 255.0
    
        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(self.size, p=0.5)
        
        img_s1_copy_ = deepcopy(img_s1_)
        img_s1_copy_ = np.asarray(img_s1_copy_)
        
        img_s1 = torch.from_numpy(np.array(img_s1)).float() / 255.0

        cutmix_box2 = obtain_cutmix_box(self.size, p=0.5)
            
        img_s2 = np.asarray(img_s2).astype(np.float32) / 255.0
        coeffs_s2 = pywt.dwt2(img_s2, 'haar')
        ll_s2, details_s2 = coeffs_s2
        
        ll_s2 = self.freq_comp_perturb(ll_s2)    
            
        list_details_s2 = list(details_s2)
            
        list_details_s2[0] = self.freq_comp_perturb(list_details_s2[0])  
        list_details_s2[1] = self.freq_comp_perturb(list_details_s2[1])        
        list_details_s2[2] = self.freq_comp_perturb(list_details_s2[2])
           
        details_s2 = tuple(list_details_s2)
        img_s2 = pywt.idwt2((ll_s2, details_s2), wavelet='haar')
            
        img_min, img_max = np.min(img_s2), np.max(img_s2)
        img_s2_png = (((img_s2 - img_min) / (img_max - img_min)) * 255).astype(np.uint8)
        img_s2_png = Image.fromarray(img_s2_png.astype(np.uint8), mode='L')
            
        img_s2_png_copy = np.asarray(img_s2_png)
            
        img_diff_2 = abs(img_s2_png_copy - img_s1_copy_)
        
        img_add_2 = img_s2_png_copy + img_diff_2 #* 2.0
            
        img_add_2 = torch.from_numpy(img_add_2).float() / 255

        return img.unsqueeze(0), img_s1.unsqueeze(0), img_add_2.unsqueeze(0), cutmix_box1, cutmix_box2
        
        
    def __len__(self):
        return len(self.ids)


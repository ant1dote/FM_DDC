import os
from pkgutil import extend_path
import json5
from tqdm import tqdm
import collections
from PIL import Image
import glob
from copy import deepcopy
import random
import numpy as np
from PIL import Image, ImageFilter
import h5py
from torchvision import transforms
import pywt
save_path = '/home/user/busi'

def blur(img, p=1.0):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img

def freq_comp_perturb(freq_comp, freq_name):
    hf_min, hf_max = np.min(freq_comp), np.max(freq_comp)
    freq_comp = (((freq_comp - hf_min) / (hf_max - hf_min)) * 255).astype(np.uint8)
    freq_comp = Image.fromarray(freq_comp)      
    freq_comp.save(save_path+'/'+freq_name+'.png')     
    freq_comp = transforms.ColorJitter(0, [0.1, 0.5], 0.0, 0.0)(freq_comp)    
    freq_comp.save(save_path+'/'+freq_name+'_jittered.png')
    freq_comp = np.array(freq_comp).astype(np.float32)
    min_blur_hf1, max_blur_hf1 = np.min(freq_comp), np.max(freq_comp)
    freq_comp = hf_min + ((freq_comp - min_blur_hf1) / (max_blur_hf1 - min_blur_hf1)) * (hf_max - hf_min)
    #freq_comp = freq_comp.transpose((2, 0, 1))
    return freq_comp

if __name__ =='__main__':
    path = '/home/user/Dataset_BUSI_with_GT/malignant/malignant (23).png'
    mask_path = '/home/user/Dataset_BUSI_with_GT/malignant/malignant (23)_mask.png'
    #path = '/home/user/ACDC/data/slices/'
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    #sample = h5py.File(path)
    #img, mask = sample['image'][:], sample['label'][:] 
    #img = Image.fromarray((img * 255).astype(np.uint8))
    img = Image.open(path)
    mask = Image.open(mask_path)
    img.save(save_path+'/img.png')
    img = img.convert(mode='L')
    img = img.resize((256, 256), Image.BICUBIC)
    mask = mask.resize((256, 256), Image.NEAREST) 
    #mask = Image.fromarray(mask * 255)
    mask.save(save_path+'/mask.png')
    #img_s1 = blur(img, p=1.0)
    #img_s1.save(save_path+'/blur.png')
    
    #if random.random()<0.8:
    img_s1 = transforms.ColorJitter(0.0, (0.1, 0.5), 0.0, 0.0)(img)
    
    img_s1.save(save_path+'/colorjitter.png')

    img_s2 = deepcopy(img)
    img_s2 = np.asarray(img_s2).astype(np.float32) / 255.0
    '''
    img_s2_fft = np.fft.fftn(img_s2)
    img_s2_fft = np.fft.fftshift(img_s2_fft)
    amp, pha = np.abs(img_s2_fft), np.angle(img_s2_fft)

    amp_min, amp_max = np.min(amp), np.max(amp)
    amp = (((amp - amp_min) / (amp_max - amp_min)) * 255).astype(np.uint8)
    amp_img = Image.fromarray(amp)
    amp_img = transforms.ColorJitter(0.0, (0.2, 0.5), 0.0, 0.0)(amp_img)
    amp_img.save(save_path+'/amp.png')

    pha_min, pha_max = np.min(pha), np.max(pha)
    pha = (((pha - pha_min) / (pha_max - pha_min)) * 255).astype(np.uint8)
    pha_img = Image.fromarray(pha)
    pha_img.save(save_path+'/pha.png')
    pha_img = transforms.ColorJitter(0.0, (0.2, 0.5), 0.0, 0.0)(pha_img)
    pha_img.save(save_path+'/pha_jitter.png')

    amp_img = np.array(amp_img).astype(np.float32)
    amp_img = amp_min + ((amp_img - np.min(amp_img)) / (np.max(amp_img) - np.min(amp_img))) * (amp_max - amp_min)

    pha_img = np.array(pha_img).astype(np.float32)
    pha_img = pha_min + ((pha_img - np.min(pha_img)) / (np.max(pha_img) - np.min(pha_img))) * (pha_max - pha_min)

    img_s2_ = amp_img * np.exp(1j * pha_img)
    img_s2_ = np.fft.ifftshift(img_s2_)
    img_s2_ = np.fft.ifftn(img_s2_)
    img_s2_ = np.abs(img_s2_)
    img_s2_min, img_s2_max = np.min(img_s2_), np.max(img_s2_)
    img_s2_ = (((img_s2_ - img_s2_min) / (img_s2_max - img_s2_min)) * 255).astype(np.uint8)
    img_s2_ = Image.fromarray(img_s2_)
    img_s2_.save(save_path+'/img_fft.png')
    img_s2_ = np.array(img_s2_)
    '''
    coeffs_s2 = pywt.dwt2(img_s2, 'haar')
    ll_s2, details_s2 = coeffs_s2
        
    ll_s2 = freq_comp_perturb(ll_s2, freq_name='lf')    
            
    list_details_s2 = list(details_s2)
            
    list_details_s2[0] = freq_comp_perturb(list_details_s2[0], freq_name='hf')  
    list_details_s2[1] = freq_comp_perturb(list_details_s2[1], freq_name='vf')        
    list_details_s2[2] = freq_comp_perturb(list_details_s2[2], freq_name='df')
           
    details_s2 = tuple(list_details_s2)
    img_s2 = pywt.idwt2((ll_s2, details_s2), wavelet='haar')
            
    img_min, img_max = np.min(img_s2), np.max(img_s2)
    img_s2_png = (((img_s2 - img_min) / (img_max - img_min)) * 255).astype(np.uint8)
    img_s2_png = Image.fromarray(img_s2_png.astype(np.uint8), mode='L')
    img_s2_png.save(save_path+'/idwt.png')
    img_s2_png = np.array(img_s2_png)
    
    img_s1_copy = deepcopy(img_s1)
    img_s1_copy = np.asarray(img_s1_copy)

    #img_s2_png_copy = np.asarray(img_s2_png)
            
    img_diff_2 = abs(img_s2_png - img_s1_copy)
    #img_fft_diff = abs(img_s2_ - img_s1_copy)
    #img_fft_add = img_s2_ + img_fft_diff
    #img_fft_add = Image.fromarray(img_fft_add)
    #img_fft_add.save(save_path+'/fft_perturb.png')
        
    img_add_2 = img_s2_png + img_diff_2 * 1.2
    img_add_ = Image.fromarray(img_add_2.astype(np.uint8))
    img_add_.save(save_path+'/fsp_1.2.png')

    img_add_1 = img_s2_png + img_diff_2 * 1.0
    img_add_1 = Image.fromarray(img_add_1.astype(np.uint8))
    img_add_1.save(save_path+'/fsp_1.png')
            
    img_add_2 = img_add_2 / 255.0
        




    a=1

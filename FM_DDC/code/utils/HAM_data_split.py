import os
from pkgutil import extend_path
import json5
from tqdm import tqdm
import collections
from PIL import Image
import glob
import random
import numpy as np
import h5py
import shutil


def get_files_with_extensions(folder_path, extension):
    file_list = []
    for file_name in glob.glob(os.path.join(folder_path, "*." +extension)):
        file_list.append(os.path.basename(file_name))
    return file_list

def find_duplicates(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    duplicates = set1 & set2
    return list(duplicates)

if __name__=='__main__':
    
    #img_path = '/home/user/HAM10000/HAM10000_images/'
    #mask_path = '/home/user/HAM10000/HAM10000_masks/'
    #txt_path = '/home/user/HAM10000/'
    path = '/home/user/HAM10000/all_files'
    img_files = os.listdir(path)

    #training_imgs = get_files_with_extensions('/home/user/HAM10000/train_3', 'h5')
    #all_images_list = 'HAM10000_6000_training_images_3.txt'
    #training_list = open(txt_path+all_images_list,'w')
    
    #for labeled_img in training_imgs:
    #    training_list.write(labeled_img + '\n')
    #training_list.close()

    txt_path = '/home/user/ham10000/'
    if not os.path.exists(txt_path):
        os.makedirs(txt_path)

    train_num = 6000
    val_num = 1015
    test_num = 3000
    
    img_size = 256

    train_samples = random.sample(img_files, train_num)
    valAndtest = set(img_files) - set(train_samples)
    val_samples = random.sample(list(valAndtest), val_num)
    test_samples = list(set(valAndtest) - set(val_samples)) 

    repeat_num = "new_3"
    sample_ratio = 0.05
    sample_num = int(len(train_samples) * sample_ratio)
    sample_labeled_img = random.sample(train_samples, sample_num)
    unlabeled_img = list(set(train_samples) - set(sample_labeled_img))
    
    leak_1 = find_duplicates(train_samples, val_samples)
    leak_2 = find_duplicates(train_samples, test_samples)
    leak_3 = find_duplicates(val_samples, test_samples)
    
    if sample_ratio == 0.05:
        labeled_list = r'5%_labeled_' + repeat_num + '.txt'
        unlabeled_list = r'5%_unlabeled_'+ repeat_num + '.txt'
    elif sample_ratio == 1.00:
        labeled_list = r'FS_labeled.txt'
    elif sample_ratio == 0.01:
        labeled_list = r'1%_labeled_' + repeat_num + '.txt'
        unlabeled_list = r'1%_unlabeled_'+ repeat_num + '.txt'
    
    
    lab_list = open(txt_path+labeled_list, 'w')
    for labeled_img in sample_labeled_img:
        lab_list.write(labeled_img + '\n')
    lab_list.close()
    
    unlab_list = open(txt_path+unlabeled_list, 'w')
    for unlabeled_img in unlabeled_img:
        unlab_list.write(unlabeled_img + '\n')
    unlab_list.close()    
    
    val_list_name = r'val_set_' + repeat_num + '.txt'
    val_list = open(txt_path + val_list_name, 'w')
    for img in val_samples:
        val_list.write(img + '\n')
    val_list.close()

    test_list_name = r'test_set_' + repeat_num + '.txt'
    test_list = open(txt_path + test_list_name, 'w')
    for img in test_samples:
        test_list.write(img + '\n')
    test_list.close()
    

    
    root_path = '/home/user/ham10000/'
    train_folder = root_path + 'train_' + repeat_num + '/' 
    val_folder = root_path + 'val_' + repeat_num + '/'
    test_folder = root_path + 'test_' + repeat_num + '/'

    folders = [train_folder, val_folder, test_folder]
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    for file in tqdm(train_samples):
        shutil.copy(path+'/'+file, train_folder+file)
    
    for file in tqdm(val_samples):
        shutil.copy(path+'/'+file, val_folder+file)

    for file in tqdm(test_samples):
        shutil.copy(path+'/'+file, test_folder+file)
    
    a=1

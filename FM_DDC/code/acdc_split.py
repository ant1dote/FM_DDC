import os
from pkgutil import extend_path
import json5
from tqdm import tqdm
import collections
from PIL import Image
import glob
import random

def get_files_with_extensions(folder_path, extension):
    file_list = []
    for file_name in glob.glob(os.path.join(folder_path, "*." +extension)):
        file_list.append(os.path.basename(file_name))
    return file_list

if __name__=='__main__':
    root_path = '/home/user/ACDC'
    train_slice_list = list() 
    train_patient_list =list()
    with open(root_path+'/train_slices.list', 'r') as r:
        lines = r.readlines()
        for line in lines:
            train_slice_list.append(line)
            train_patient_list.append(line.split('_')[0])
    train_patient = list(set(train_patient_list))
    labeled_patients_num = 1
    labeled_patients = random.sample(train_patient, labeled_patients_num)
    unlabled_patients = list(set(train_patient) - set(labeled_patients))
    
    labeled_text = open(root_path+'/1_labeled_2.txt', 'w')
    unlabeled_text = open(root_path+'/1_unlabeled_2.txt', 'w')

    for lab_patient in labeled_patients:
        for slice in train_slice_list:
            if lab_patient in slice:
                labeled_text.write('/data/slices/'+slice.split('\n')[0]+'.h5\n')
            else:
                unlabeled_text.write('/data/slices/'+slice.split('\n')[0]+'.h5\n')
    labeled_text.close()
    unlabeled_text.close()

    a=1
    
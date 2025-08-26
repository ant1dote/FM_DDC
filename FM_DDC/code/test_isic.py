import argparse
import os
import shutil

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
import PIL.Image as Image
from networks.net_factory import net_factory
from networks.net_factory import ViTs
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/home/user/ISIC2018', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='Dual_Domain_Match_HAM10000_1%_RE1', help='experiment_name')
parser.add_argument('--model', type=str, default='attunet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=2, help='output channel of network')
parser.add_argument('--cfg', type=str, default='/home/user/Frequency_Matters_DDC_SSL_Skin_Lesion_Segmentation/code/configs/swin_tiny_patch4_window7_224_lite.yaml', help='path to config file', )
parser.add_argument('--labelnum', type=int, default=20, help='labeled data')
parser.add_argument('--batch_size', type=int, default=1, help='batch_size per gpu')
parser.add_argument('--patch_size', type=list,  default=[224, 224], help='patch size of network input')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--dataset', type = str, default = 'HAM10000')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

def calculate_metric_percase(pred, gt):
    
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    asd = metric.binary.asd(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)

    return dice, jc, hd95, asd


def test_single_volume(case, net, test_save_path, FLAGS):
    
    h5f = h5py.File(FLAGS.root_path + "/test_h5/{}".format(case.split('.')[0]+'.h5'), 'r')
    slice, label = h5f['image'][:], h5f['label'][:]
    prediction = np.zeros_like(label)
    
    x, y = slice.shape[1], slice.shape[2]
    slice = zoom(slice, (1, 224 / x, 224 / y), order=0)
    label = zoom(label, (224 / x, 224 / y), order=0)
    input = torch.from_numpy(slice).unsqueeze(0).float().cuda()
    net.eval()
    with torch.no_grad():
        out_main = net(input)
        if len(out_main)>1:
            out_main=out_main[0]
        out = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        #pred = zoom(out, (x / 256, y / 256), order=0)
        prediction = out
    if np.sum(prediction == 1)==0:
        first_metric = 0,0,0,0
    else:
        first_metric = calculate_metric_percase(prediction == 1, label == 1)
    '''
    img_name = FLAGS.dataset + '_' +case.split('_')[1] + '.png'
    gt_name = FLAGS.dataset + '_' +case.split('_')[1] + '_gt.png'
    pred_name = FLAGS.dataset + '_' +case.split('_')[1] + '_pred.png'
    slice_ = slice.transpose((1, 2, 0)) * 255
    img = Image.fromarray(slice_.astype(np.uint8))
    img.save(test_save_path + img_name)
    label = 1 - label
    label_ = Image.fromarray(label * 255)
    label_.save(test_save_path + gt_name, mode='F')
    prediction_ = prediction.astype(np.uint8) * 255
    prediction = Image.fromarray(prediction_)
    prediction.save(test_save_path + pred_name)
    '''

    '''
    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    '''
    return first_metric


def Inference(FLAGS):
    with open(FLAGS.root_path + '/test_set.txt', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0] for item in image_list])
    snapshot_path = "./model/ACDC_{}_{}_labeled/{}".format(FLAGS.exp, FLAGS.labelnum, FLAGS.model)
    test_save_path = "/home/user/MC-Net-main/predictions/{}_{}_labeled/".format(FLAGS.exp, FLAGS.labelnum)
    
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)
    
    args = parser.parse_args()
    net = net_factory(args, net_type = FLAGS.model, in_chns=3, class_num=FLAGS.num_classes)
    #save_model_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    #config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    #config_vit.n_classes = args.num_classes
    #config_vit.n_skip = 3
    #net = ViTs(config_vit, img_size=256, num_classes=config_vit.n_classes).cuda()
    
    save_model_path = '/media/user/SX5PRO/ISIC2018_FSP_attUNet_RE1_25_labeled/attunet_best_model.pth'
    net.load_state_dict(torch.load(save_model_path), strict=True)
    print("init weight from {}".format(save_model_path))
    net.eval()
    first_total = 0.0
    
    for case in tqdm(image_list):
        first_metric = test_single_volume(case, net, test_save_path, FLAGS)
        first_total += np.asarray(first_metric)
    avg_metric = [first_total / len(image_list)]
    return avg_metric, test_save_path

if __name__ == '__main__':
    FLAGS = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    metric, test_save_path = Inference(FLAGS)
    print(metric)
    print(metric[0])
    with open(test_save_path+'../performance.txt', 'w') as f:
        f.writelines('metric is {} \n'.format(metric))
        f.writelines('average metric is {}\n'.format(metric[0]))
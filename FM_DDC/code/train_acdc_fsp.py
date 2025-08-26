import argparse
import logging
import os
import random
import shutil
import sys
import numpy as np
import torch
import torch.backends.cudnn as  cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss
import torch.nn as nn
from tqdm import tqdm
from skimage.measure import label
from dataloaders.acdc import ACDC_DIFF_FSP
from networks.net_factory import net_factory
from utils import losses, val_2d


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/home/user/ACDC', help='Name of Experiment')
parser.add_argument('--dataset', type=str, default = 'ACDC', help='the dataset used in the experiments')

parser.add_argument('--labeled-id-path', type=str, default='/home/user/ACDC/3_labeled_2.txt')
parser.add_argument('--unlabeled-id-path', type=str, default='/home/user/ACDC/3_unlabeled_2.txt')

parser.add_argument('--exp', type=str, default=r'fsp_Coif2Wavelet_RE2', help='experiment_name')
parser.add_argument('--model', type=str, default='unets_noa', help='model_name')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=12, help='batch_size per gpu')

parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')

parser.add_argument('--patch_size', type=list,  default=[256, 256], help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4, help='output channel of network')

#label and unlabel
parser.add_argument('--labeled_bs', type=int, default=12, help='labeled_batch_size per gpu')
parser.add_argument('--labelnum', type=int, default=3, help='labeled data')
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
# costs
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')

args = parser.parse_args()

dice_loss = losses.DiceLoss(n_classes=args.num_classes)

def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])

def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])

def save_net_opt(net, optimizer, path):
    state = {
        'net':net.state_dict(),
        'opt':optimizer.state_dict(),
    }
    torch.save(state, str(path))

def get_ACDC_LargestCC(segmentation):
    class_list = []
    for i in range(1, 4):
        temp_prob = segmentation == i * torch.ones_like(segmentation)
        temp_prob = temp_prob.detach().cpu().numpy()
        labels = label(temp_prob)
        # -- with 'try'
        assert(labels.max() != 0)  # assume at least 1 CC
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        class_list.append(largestCC * i)
    acdc_largestCC = class_list[0] + class_list[1] + class_list[2]
    return torch.from_numpy(acdc_largestCC).cuda()

def get_ACDC_2DLargestCC(segmentation):
    batch_list = []
    N = segmentation.shape[0]
    for i in range(0, N):
        class_list = []
        for c in range(1, 4):
            temp_seg = segmentation[i] #== c *  torch.ones_like(segmentation[i])
            temp_prob = torch.zeros_like(temp_seg)
            temp_prob[temp_seg == c] = 1
            temp_prob = temp_prob.detach().cpu().numpy()
            labels = label(temp_prob)          
            if labels.max() != 0:
                largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
                class_list.append(largestCC * c)
            else:
                class_list.append(temp_prob)
        
        n_batch = class_list[0] + class_list[1] + class_list[2]
        batch_list.append(n_batch)

    return torch.Tensor(batch_list).cuda()
    

def get_ACDC_masks(output, nms=0):
    probs = F.softmax(output, dim=1)
    _, probs = torch.max(probs, dim=1)
    if nms == 1:
        probs = get_ACDC_2DLargestCC(probs)      
    return probs


def self_train(args, snapshot_path):
    
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
     
    model = net_factory(args, net_type= args.model, in_chns=1, class_num=num_classes)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainset_u = ACDC_DIFF_FSP(args.dataset, args.root_path, 'train_u', 
                             args.patch_size[0], args.unlabeled_id_path)
    trainset_l = ACDC_DIFF_FSP(args.dataset, args.root_path, 'train_l', 
                             args.patch_size[0],args.labeled_id_path, nsample=len(trainset_u.ids))
    
    db_val = ACDC_DIFF_FSP(args.dataset, args.root_path, "val")
    
    trainloader_l = DataLoader(trainset_l, batch_size=args.batch_size, 
                               pin_memory=True, num_workers=1, drop_last=True, worker_init_fn=worker_init_fn)
    trainloader_u = DataLoader(trainset_u, batch_size=args.batch_size,
                              pin_memory=True, num_workers=1, drop_last=True, worker_init_fn=worker_init_fn)
    trainloader_u_mix = DataLoader(trainset_u, batch_size=args.batch_size,
                                  pin_memory=True, num_workers=1, drop_last=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, pin_memory=True, num_workers=1, drop_last=False)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start self_training")
    #logging.info("{} iterations per epoch".format(len(trainloader)))

    model.train()
    
    ce_loss = nn.CrossEntropyLoss()

    iter_num = 0
    max_epoch = max_iterations // (args.batch_size-1) + 1
    best_performance = 0.0
    best_hd = 100
    iterator = tqdm(range(max_epoch), ncols=70)
    
    for _ in iterator:
        loader = zip(trainloader_l, trainloader_u, trainloader_u_mix)
        for i,((img_x, mask_x),
               (img_u_w, img_u_s1, img_u_s2, cutmix_box1, cutmix_box2),
               (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, _, _)) in enumerate(loader):
        
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w = img_u_w.cuda()
            img_u_s1, img_u_s2 = img_u_s1.cuda(), img_u_s2.cuda()
            cutmix_box1= cutmix_box1.cuda()
            img_u_w_mix = img_u_w_mix.cuda()
            img_u_s1_mix, img_u_s2_mix = img_u_s1_mix.cuda().float(), img_u_s2_mix.cuda().float()

            with torch.no_grad():
                pred_u_w_mix_a, pred_u_w_mix_b = model(img_u_w_mix)
                mask_u_w_mix_a, mask_u_w_mix_b = get_ACDC_masks(pred_u_w_mix_a, nms=1), get_ACDC_masks(pred_u_w_mix_b, nms=1)

            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape)==1]= \
                img_u_s1_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape)==1]
            img_u_s2[cutmix_box1.unsqueeze(1).expand(img_u_s2.shape)==1]= \
                img_u_s2_mix[cutmix_box1.unsqueeze(1).expand(img_u_s2.shape)==1]

            preds_a = model.subnet1(torch.cat((img_x, img_u_w)))
            preds_b = model.subnet2(torch.cat((img_x, img_u_w)))
            
            predsa_x, predsa_u_w = preds_a[:args.labeled_bs], preds_a[args.labeled_bs:]
            predsb_x, predsb_u_w = preds_b[:args.labeled_bs], preds_b[args.labeled_bs:]
            
            pred_u_w_fp_a = model.subnet1(img_u_w, feat_perb=True, freq_perb=False)
            pred_u_w_fp_b = model.subnet2(img_u_w, feat_perb=False, freq_perb=True)
            
            predsa_u_s1 = model.subnet1(img_u_s1)
            predsb_u_s2 = model.subnet2(img_u_s2)
            
            predsa_u_s1_soft = F.softmax(predsa_u_s1, dim=1)
            predsb_u_s2_soft = F.softmax(predsb_u_s2, dim=1)
            
            with torch.no_grad():    
                maska_u_w, maskb_u_w = get_ACDC_masks(predsa_u_w, nms=1), get_ACDC_masks(predsb_u_w, nms=1)
            
            maska_u_w_cutmixed1 = maska_u_w.clone()
            maskb_u_w_cutmixed1 = maskb_u_w.clone()
            maska_u_w_cutmixed1[cutmix_box1==1] = mask_u_w_mix_a[cutmix_box1==1]
            maskb_u_w_cutmixed1[cutmix_box1==1] = mask_u_w_mix_b[cutmix_box1==1] 
            
            loss_ce_sup = (ce_loss(predsa_x, mask_x) + ce_loss(predsb_x, mask_x)  ) / 2.0
            loss_dice_sup = (dice_loss(F.softmax(predsa_x, dim=1), mask_x.unsqueeze(1).float()) + dice_loss(F.softmax(predsb_x, dim=1), mask_x.unsqueeze(1).float())) / 2.0
            loss_sup = (loss_ce_sup + loss_dice_sup) / 2.0
            
            del predsa_x, predsb_x, mask_x

            loss_ce_unsup_a = (ce_loss(predsa_u_s1, maskb_u_w_cutmixed1.long()) ) 
            loss_ce_unsup_b = (ce_loss(predsb_u_s2, maska_u_w_cutmixed1.long())) 
            loss_ce_unsup = (loss_ce_unsup_a + loss_ce_unsup_b) / 2.0

            loss_dice_unsup_a = (dice_loss(F.softmax(predsa_u_s1, dim=1), maskb_u_w_cutmixed1.unsqueeze(1).float())) 
            loss_dice_unsup_b = (dice_loss(F.softmax(predsb_u_s2, dim=1), maska_u_w_cutmixed1.unsqueeze(1).float())) 
            loss_dice_unsup = (loss_dice_unsup_a + loss_dice_unsup_b) / 2.0
            loss_unsup = (loss_ce_unsup + loss_dice_unsup) / 2.0

            del predsa_u_s1, predsb_u_s2, maska_u_w_cutmixed1, maskb_u_w_cutmixed1

            loss_fp_a = (ce_loss(pred_u_w_fp_a, maskb_u_w.long())+ dice_loss(F.softmax(pred_u_w_fp_a, dim=1), maskb_u_w.unsqueeze(1).float())) / 2.0
            loss_fp_b = (ce_loss(pred_u_w_fp_b, maska_u_w.long())+ dice_loss(F.softmax(pred_u_w_fp_b, dim=1), maska_u_w.unsqueeze(1).float())) / 2.0

            loss_con = (losses.mse_loss(F.softmax(pred_u_w_fp_a, dim=1), F.softmax(pred_u_w_fp_b, dim=1)) \
                        + losses.mse_loss(predsa_u_s1_soft, predsb_u_s2_soft)) / 2.0
            

            loss_fp = (loss_fp_a + loss_fp_b) / 2.0
            
            loss = loss_sup + loss_unsup + loss_fp + loss_con
            #loss = (0.6*loss_sup + 0.1 * loss_unsup + 0.1 * loss_con + 0.2 * loss_fp) #/ 2.0
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

            #lr = base_lr * (1 - iter_num / max_iterations) ** 0.9
            #optimizer.param_groups[0]["lr"] = lr
            iter_num += 1
            
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/mix_dice', loss_sup, iter_num)
            writer.add_scalar('info/mix_ce', loss_unsup, iter_num)
            writer.add_scalar('info/mix_ce', loss_fp, iter_num)
            #writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)     

            logging.info('iteration %d: loss: %f, loss_sup: %f, loss_unsup: %f, loss_fp: %f, loss_con: %f'%(iter_num, loss, loss_sup, loss_unsup, loss_fp, loss_con))
                
            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, (img, mask) in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(img, mask, model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1), metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)

                logging.info('iteration %d : mean_dice : %f' % (iter_num, performance))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()

if __name__ == "__main__":

    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    # -- path to save models
    
    self_snapshot_path = "/home/user/Frequency_Matters_DDC_SSL_Skin_Lesion_Segmentation/new_discussions/{}_{}_{}_labeled/".format(args.dataset,args.exp, args.labelnum)
   
    if not os.path.exists(self_snapshot_path):
        os.makedirs(self_snapshot_path)
    shutil.copy(str(sys.argv[0]), self_snapshot_path)
    if os.path.exists(self_snapshot_path + '/code'):
        shutil.rmtree(self_snapshot_path + '/code')
    #shutil.copytree('/home/user/Frequency_Matters_DDC_SSL_Skin_Lesion_Segmentation/', self_snapshot_path + '/code',shutil.ignore_patterns(['.git', '__pycache__']))
   
    logging.basicConfig(filename=self_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args, self_snapshot_path)
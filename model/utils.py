# -*- coding: utf-8 -*-
"""
@Time ： 2024/7/12 17:37
@Auth ： 归去来兮
@File ：utils.py
@IDE ：PyCharm
@Motto:花中自幼微风起
"""
import numpy   as np
from   PIL     import Image
from   skimage import measure
from PIL import Image, ImageOps, ImageFilter
import platform, os
from torch.utils.data.dataset import Dataset
import random
import numpy as np
import  torch
from torch.nn import init
from datetime import datetime
import torch.nn
import argparse
import shutil
from  matplotlib  import pyplot as plt
#from parse_args_mytrain import parse_args
import os.path as osp
import cv2
def load_image_img(srcpath):
    img=cv2.imdecode(np.fromfile(srcpath, dtype=np.uint8), -1)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
def load_image(srcpath):
    img=cv2.imdecode(np.fromfile(srcpath, dtype=np.uint8), -1)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # if img.ndim == 3:
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = img / 255
    # img = cv2.imread(srcpath, 0)
    return img
def load_dataset ():
    args=parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    train_txt = args.train_dataset_dir+args.split_method+args.train_txt
    test_txt  = args.test_dataset_dir+args.split_method+args.test_txt
    train_img_ids = []
    test_img_ids = []
    with open(train_txt, "r") as f:
        line = f.readline()
        while line:
            train_img_ids.append(line.split('\n')[0])
            line = f.readline()
        f.close()
    with open(test_txt, "r") as f:
        line = f.readline()
        while line:
            test_img_ids.append(line.split('\n')[0])
            line = f.readline()
        f.close()
    return train_img_ids,test_img_ids,test_txt

class TrainSetLoader(Dataset):

    """Iceberg Segmentation dataset."""
    NUM_CLASS = 1

    def __init__(self, dataset_dir, img_id ,base_size=512,crop_size=480,transform=None,suffix='.png'):
        super(TrainSetLoader, self).__init__()
        self.dataset_dir = dataset_dir
        self.transform   = transform
        self._items = img_id
        self.masks  = dataset_dir+'/'+'masks'
        self.images = dataset_dir+'/'+'images'

        self.base_size   = base_size
        self.crop_size   = crop_size
        self.suffix      = suffix
        self.aug = augumentation()


    def _sync_transform(self, img, mask, img_id):
        # random mirror
        if random.random() < 0.5:
            img   = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask  = mask.transpose(Image.FLIP_LEFT_RIGHT)
        # if random.random() < 0.5:
        #     img = img[::-1, :]
        #     mask = mask[::-1, :]
        # if random.random() < 0.5:
        #     img = img[:, ::-1]
        #     mask = mask[:, ::-1]



        crop_size = self.crop_size
        # random scale (short edge)
        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img  = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img  = ImageOps.expand(img,  border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1   = random.randint(0, w - crop_size)
        y1   = random.randint(0, h - crop_size)
        img  = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        img, mask = np.array(img), np.array(mask, dtype=np.float32)


        return img, mask

    def __getitem__(self, idx):

        img_id     = self._items[idx]                      # idx：('../SIRST', 'Misc_70') 成对出现，因为我的workers设置为了2
        img_path   = self.images+'/'+img_id+self.suffix    # img_id的数值正好补了self._image_path在上面定义的2个空
        label_path = self.masks +'/'+img_id+self.suffix

        img  = Image.open(img_path).convert('RGB')         ##由于输入的三通道、单通道图像都有，所以统一转成RGB的三通道，这也符合Unet等网络的期待尺寸
        mask = Image.open(label_path)


        # synchronized transform
        img, mask = self._sync_transform(img, mask, img_id)

        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)

        mask = np.expand_dims(mask[:,:,0] if len(np.shape(mask))>2 else mask, axis=0).astype('float32')/ 255.0

        return img, torch.from_numpy(mask)  #img_id[-1]

    def __len__(self):
        return len(self._items)


class TestSetLoader(Dataset):
    """Iceberg Segmentation dataset."""
    NUM_CLASS = 1

    def __init__(self, dataset_dir, img_id, transform=None, base_size=512, crop_size=480, suffix='.png'):
        super(TestSetLoader, self).__init__()
        self.transform = transform
        self._items    = img_id
        self.masks     = dataset_dir+'/'+'masks'
        self.images    = dataset_dir+'/'+'images'
        self.base_size = base_size
        self.crop_size = crop_size
        self.suffix    = suffix

    def _testval_sync_transform(self, img, mask):
        base_size = self.base_size
        img  = img.resize ((base_size, base_size), Image.BILINEAR)
        mask = mask.resize((base_size, base_size), Image.NEAREST)


        # final transform
        img, mask = np.array(img), np.array(mask, dtype=np.float32)  # images: <class 'mxnet.ndarray.ndarray.NDArray'> (512, 512, 3)
        return img, mask

    def __getitem__(self, idx):
        # print('idx:',idx)
        img_id     = self._items[idx]  # idx：('../SIRST', 'Misc_70') 成对出现，因为我的workers设置为了2
        img_path   = self.images + '/' + img_id + self.suffix    # img_id的数值正好补了self._image_path在上面定义的2个空
        label_path = self.masks  + '/' + img_id + self.suffix

        img  = Image.open(img_path).convert('RGB')  ##由于输入的三通道、单通道图像都有，所以统一转成RGB的三通道，这也符合Unet等网络的期待尺寸
        mask = Image.open(label_path)

        w, h = img.size
        size = [w, h]

        # synchronized transform
        img, mask = self._testval_sync_transform(img, mask)

        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)

        mask = np.expand_dims(mask[:,:,0] if len(np.shape(mask))>2 else mask, axis=0).astype('float32')/ 255.0


        return img, torch.from_numpy(mask)  # img_id[-1]

    def __len__(self):
        return len(self._items)


###初始化使用
def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        #init.xavier_uniform(m.weight.data)
        init.kaiming_normal_(m.weight,nonlinearity='relu')
       # init.kaiming_uniform_(m.weight,nonlinearity='leaky_relu')
def weights_init_xavier_t(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        # init.xavier_uniform(m.weight.data)
        init.kaiming_normal_(m.weight,nonlinearity='relu')
        # init.kaiming_uniform_(m.weight,nonlinearity='relu')
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def SoftIoULoss(pred, target):
        # Old One
        pred = torch.sigmoid(pred)
        smooth = 1

        # print("pred.shape: ", pred.shape)
        # print("target.shape: ", target.shape)

        intersection = pred * target
        loss = (intersection.sum() + smooth) / (pred.sum() + target.sum() - intersection.sum() + smooth)

        # loss = (intersection.sum(axis=(1, 2, 3)) + smooth) / \
        #        (pred.sum(axis=(1, 2, 3)) + target.sum(axis=(1, 2, 3))
        #         - intersection.sum(axis=(1, 2, 3)) + smooth)

        loss = 1 - loss.mean()
        # loss = (1 - loss).mean()

        return loss

class ROCMetric():
    """Computes pixAcc and mIoU metric scores
    """
    def __init__(self, nclass, bins):  #bin的意义实际上是确定ROC曲线上的threshold取多少个离散值
        super(ROCMetric, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.tp_arr = np.zeros(self.bins+1)
        self.pos_arr = np.zeros(self.bins+1)
        self.fp_arr = np.zeros(self.bins+1)
        self.neg_arr = np.zeros(self.bins+1)
        self.class_pos=np.zeros(self.bins+1)
        # self.reset()

    def update(self, preds, labels):
        for iBin in range(self.bins+1):
            score_thresh = (iBin + 0.0) / self.bins
            # print(iBin, "-th, score_thresh: ", score_thresh)
            i_tp, i_pos, i_fp, i_neg,i_class_pos = cal_tp_pos_fp_neg(preds, labels, self.nclass,score_thresh)
            self.tp_arr[iBin]   += i_tp
            self.pos_arr[iBin]  += i_pos
            self.fp_arr[iBin]   += i_fp
            self.neg_arr[iBin]  += i_neg
            self.class_pos[iBin]+=i_class_pos

    def get(self):

        tp_rates    = self.tp_arr / (self.pos_arr + 0.001)
        fp_rates    = self.fp_arr / (self.neg_arr + 0.001)

        recall      = self.tp_arr / (self.pos_arr   + 0.001)
        precision   = self.tp_arr / (self.class_pos + 0.001)


        return tp_rates, fp_rates, recall, precision

    def reset(self):

        self.tp_arr   = np.zeros([11])
        self.pos_arr  = np.zeros([11])
        self.fp_arr   = np.zeros([11])
        self.neg_arr  = np.zeros([11])
        self.class_pos= np.zeros([11])

def cal_tp_pos_fp_neg(output, target, nclass, score_thresh):

    predict = (torch.sigmoid(output) > score_thresh).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    intersection = predict * ((predict == target).float())

    tp = intersection.sum()
    fp = (predict * ((predict != target).float())).sum()
    tn = ((1 - predict) * ((predict == target).float())).sum()
    fn = (((predict != target).float()) * (1 - predict)).sum()
    pos = tp + fn
    neg = fp + tn
    class_pos= tp+fp

    return tp, pos, fp, neg, class_pos

class PD_FA():
    def __init__(self, nclass, bins, crop_size):
        super(PD_FA, self).__init__()
        self.nclass    = nclass
        self.bins      = bins
        self.image_area_total = []
        self.image_area_match = []
        self.FA        = np.zeros(self.bins+1)
        self.PD        = np.zeros(self.bins + 1)
        self.target    = np.zeros(self.bins + 1)
        self.crop_size = crop_size
    def update(self, preds, labels):

        for iBin in range(self.bins+1):
            score_thresh = iBin * (255/self.bins)
            predits  = np.array((preds > score_thresh).cpu()).astype('int64')
            predits  = np.reshape (predits,  (self.crop_size,self.crop_size))
            labelss  = np.array((labels).cpu()).astype('int64') # P
            labelss  = np.reshape (labelss , (self.crop_size,self.crop_size))

            image = measure.label(predits, connectivity=2)
            coord_image = measure.regionprops(image)
            label = measure.label(labelss , connectivity=2)
            coord_label = measure.regionprops(label)

            self.target[iBin]    += len(coord_label)
            self.image_area_total = []
            self.image_area_match = []
            self.distance_match   = []
            self.dismatch         = []

            for K in range(len(coord_image)):
                area_image = np.array(coord_image[K].area)
                self.image_area_total.append(area_image)

            for i in range(len(coord_label)):
                centroid_label = np.array(list(coord_label[i].centroid))
                for m in range(len(coord_image)):
                    centroid_image = np.array(list(coord_image[m].centroid))
                    distance = np.linalg.norm(centroid_image - centroid_label)
                    area_image = np.array(coord_image[m].area)
                    if distance < 3:
                        self.distance_match.append(distance)
                        self.image_area_match.append(area_image)

                        del coord_image[m]
                        break

            self.dismatch = [x for x in self.image_area_total if x not in self.image_area_match]
            self.FA[iBin]+=np.sum(self.dismatch)
            self.PD[iBin]+=len(self.distance_match)

    def update_fixT(self, preds, labels,score_thresh):

        for iBin in range(self.bins+1):
            predits  = np.array((preds > score_thresh).cpu()).astype('int64')
            predits  = np.reshape (predits,  (self.crop_size,self.crop_size))
            labelss  = np.array((labels).cpu()).astype('int64') # P
            labelss  = np.reshape (labelss , (self.crop_size,self.crop_size))

            image = measure.label(predits, connectivity=2)
            coord_image = measure.regionprops(image)
            label = measure.label(labelss , connectivity=2)
            coord_label = measure.regionprops(label)

            self.target[iBin]    += len(coord_label)
            self.image_area_total = []
            self.image_area_match = []
            self.distance_match   = []
            self.dismatch         = []

            for K in range(len(coord_image)):
                area_image = np.array(coord_image[K].area)
                self.image_area_total.append(area_image)

            for i in range(len(coord_label)):
                centroid_label = np.array(list(coord_label[i].centroid))
                for m in range(len(coord_image)):
                    centroid_image = np.array(list(coord_image[m].centroid))
                    distance = np.linalg.norm(centroid_image - centroid_label)
                    area_image = np.array(coord_image[m].area)
                    if distance < 3:
                        self.distance_match.append(distance)
                        self.image_area_match.append(area_image)

                        del coord_image[m]
                        break

            self.dismatch = [x for x in self.image_area_total if x not in self.image_area_match]
            self.FA[iBin]+=np.sum(self.dismatch)
            self.PD[iBin]+=len(self.distance_match)

    def get(self,img_num,crop_size):

        Final_FA =  self.FA / ((crop_size * crop_size) * img_num)
        Final_PD =  self.PD /self.target

        return Final_FA,Final_PD


    def reset(self):
        self.FA  = np.zeros([self.bins+1])
        self.PD  = np.zeros([self.bins+1])


class mIoU():
    def __init__(self, nclass):
        super(mIoU, self).__init__()
        self.nclass = nclass
        self.reset()

    def update(self, preds, labels):
        # print('come_ininin')

        correct, labeled = batch_pix_accuracy(preds, labels)
        inter, union = batch_intersection_union(preds, labels, self.nclass)
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union


    def get(self):

        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return pixAcc, mIoU

    def reset(self):

        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0

def batch_pix_accuracy(output, target):

    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    assert output.shape == target.shape, "Predict and Label Shape Don't Match"
    predict = (output > 0).float()
    pixel_labeled = (target > 0).float().sum()
    pixel_correct = (((predict == target).float())*((target > 0)).float()).sum()



    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass):

    mini = 1
    maxi = 1
    nbins = 1
    predict = (output > 0).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")
    intersection = predict * ((predict == target).float())

    area_inter, _  = np.histogram(intersection.cpu(), bins=nbins, range=(mini, maxi))
    area_pred,  _  = np.histogram(predict.cpu(), bins=nbins, range=(mini, maxi))
    area_lab,   _  = np.histogram(target.cpu(), bins=nbins, range=(mini, maxi))
    area_union     = area_pred + area_lab - area_inter

    assert (area_inter <= area_union).all(), \
        "Error: Intersection area should be smaller than Union area"
    return area_inter, area_union

def save_model(best_iou, save_dir, save_prefix, train_loss, test_loss, recall, precision, epoch, net):
        save_mIoU_dir = 'result_WS/' + '_best_IoU_IoU.log'
        save_other_metric_dir = 'result_WS/' + '_best_IoU_other_metric.log'
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        save_model_and_result(dt_string, epoch, train_loss, test_loss, best_iou,
                              recall, precision, save_mIoU_dir, save_other_metric_dir)
        save_ckpt({
            'epoch': epoch,
            'state_dict': net,
            'loss': test_loss,
            'mean_IOU': best_iou,
        }, save_path='result_WS/' + save_dir,
            filename='model_weight' + '.pth.tar')


def save_model_and_result(dt_string, epoch, train_loss, test_loss, best_iou, recall, precision, save_mIoU_dir, save_other_metric_dir):

    with open(save_mIoU_dir, 'a') as f:
        f.write('{} - {:04d}:\t - train_loss: {:04f}:\t - test_loss: {:04f}:\t mIoU {:.4f}\n' .format(dt_string, epoch,train_loss, test_loss, best_iou))
    with open(save_other_metric_dir, 'a') as f:
        f.write(dt_string)
        f.write('-')
        f.write(str(epoch))
        f.write('\n')
        f.write('Recall-----:')
        for i in range(len(recall)):
            f.write('   ')
            f.write(str(round(recall[i], 8)))
            f.write('   ')
        f.write('\n')

        f.write('Precision--:')
        for i in range(len(precision)):
            f.write('   ')
            f.write(str(round(precision[i], 8)))
            f.write('   ')
        f.write('\n')

def save_ckpt(state, save_path, filename):
    torch.save(state, os.path.join(save_path,filename))



def save_resize_pred(pred, size, crop_size, target_image_path, val_img_ids, num, suffix):

    preds = np.array((pred > 0).cpu()).astype('int64') * 255
    preds = np.uint8(preds)

    preds = Image.fromarray(preds.reshape(crop_size, crop_size))
    # img = Image.fromarray(preds)
    img = preds.resize((size[0].item(), size[1].item()), Image.NEAREST)
    img.save(target_image_path + '/' + '%s' % (val_img_ids[num]) + suffix)

def save_Pred_GT_for_split_evalution(pred, labels, target_image_path, val_img_ids, num, suffix, crop_size):

    predsss = np.array((pred > 0).cpu()).astype('int64') * 255
    predsss = np.uint8(predsss)

    #img = Image.fromarray(predsss.reshape(crop_size, crop_size))
    img = Image.fromarray(predsss)
    img.save(target_image_path + '/' + '%s' % (val_img_ids[num]) +suffix)




class InferenceSetLoader(Dataset):
    NUM_CLASS = 1

    def __init__(self, dataset_dir, img_id, transform=None, base_size=512, crop_size=480, suffix='.png'):
        super(InferenceSetLoader, self).__init__()
        self.transform = transform
        self._items    = img_id
        #self.masks     = dataset_dir+'/'+'masks'
        self.images    = dataset_dir+'/'+'images'
        self.base_size = base_size
        self.crop_size = crop_size
        self.suffix    = suffix

    def _testval_sync_transform(self, img):
        base_size = self.base_size
        img  = img.resize ((base_size, base_size), Image.BILINEAR)
        #mask = mask.resize((base_size, base_size), Image.NEAREST)


        # final transform
        #img, mask = np.array(img), np.array(mask, dtype=np.float32)  # images: <class 'mxnet.ndarray.ndarray.NDArray'> (512, 512, 3)
        #return img, mask
        img = np.array(img)
        return img


    def __getitem__(self, idx):
        # print('idx:',idx)
        img_id     = self._items[idx]  # idx：('../SIRST', 'Misc_70') 成对出现，因为我的workers设置为了2
        img_path   = self.images + '/' + img_id + self.suffix    # img_id的数值正好补了self._image_path在上面定义的2个空
        #label_path = self.masks  + '/' + img_id + self.suffix

        img  = Image.open(img_path).convert('RGB')  ##由于输入的三通道、单通道图像都有，所以统一转成RGB的三通道，这也符合Unet等网络的期待尺寸
        #mask = Image.open(label_path)

        w, h = img.size
        size = [w, h]

        # synchronized transform
        #img, mask = self._testval_sync_transform(img, mask)
        img = self._testval_sync_transform(img)

        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)

        #mask = np.expand_dims(mask[:,:,0] if len(np.shape(mask))>2 else mask, axis=0).astype('float32')/ 255.0

        #return img, torch.from_numpy(mask), size  # img_id[-1]
        return img, size

    def __len__(self):
        return len(self._items)


class MDFA(object):
    def __init__(self, base_dir='../data/MDFA/', mode='test'):
        assert mode in ['trainval', 'test']
        if mode == 'trainval':
            self.img_dir = osp.join(base_dir, 'training')
            self.mask_dir = osp.join(base_dir, 'training')
            self.length = 9978
        elif mode == 'test':
            self.img_dir = osp.join(base_dir, 'test_org')
            self.mask_dir = osp.join(base_dir, 'test_gt')
            self.length = 100
        else:
            raise NotImplementedError

        self.mode = mode

    def __getitem__(self, i):
        if self.mode == 'trainval':
            img_path = osp.join(self.img_dir, '%06d_1.png' % i)
            mask_path = osp.join(self.mask_dir, '%06d_2.png' % i)
        elif self.mode == 'test':
            img_path = osp.join(self.img_dir, '%05d.png' % i)
            mask_path = osp.join(self.mask_dir, '%05d.png' % i)
        else:
            raise NotImplementedError

        img = load_image(img_path)
        mask = load_image(mask_path)
        return img, mask

    def __len__(self):
        return self.length


class SIRST(object):
    def __init__(self, base_dir='sirst-master/', transform=None,mode='train'):
        if mode == 'train':
            txtfile = 'train.txt'
        elif mode == 'test':
            txtfile = 'test.txt'
        else:
            raise NotImplementedError

        self.list_dir = osp.join(base_dir, 'idx_427', txtfile)
        self.imgs_dir = osp.join(base_dir, 'images')
        self.label_dir = osp.join(base_dir, 'masks')
        self.transform = transform
        self.mode = mode
        self.names = []
        self.base_size = 512
        with open(self.list_dir, 'r') as f:
            self.names += [line.strip() for line in f.readlines()]

    def _testval_sync_transform(self, img, mask):
        base_size = self.base_size
        img = img.resize((base_size, base_size), Image.BILINEAR)
        mask = mask.resize((base_size, base_size), Image.NEAREST)

        # final transform
        img, mask = np.array(img), np.array(mask,dtype=np.float32)  # images: <class 'mxnet.ndarray.ndarray.NDArray'> (512, 512, 3)
        return img, mask
    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.imgs_dir, name+'.png')
        label_path = osp.join(self.label_dir, name+'_pixels0.png')
        img = Image.open(img_path).convert('RGB')  ##由于输入的三通道、单通道图像都有，所以统一转成RGB的三通道，这也符合Unet等网络的期待尺寸
        mask = Image.open(label_path)
        w, h = img.size
        size = [w, h]
        if self.mode == "test":
        # synchronized transform
            img, mask = self._testval_sync_transform(img, mask)
        if self.transform is not None:
            img = self.transform(img)
            # mask = self.transform(mask)
        mask = np.expand_dims(mask[:, :, 0] if len(np.shape(mask)) > 2 else mask, axis=0).astype('float32') / 255.0

        return img, torch.from_numpy(mask)

    def __len__(self):
        return len(self.names)


class augumentation(object):
    def __call__(self, input, target):
        if random.random()<0.5:
            input = input[::-1, :]
            target = target[::-1, :]
        if random.random()<0.5:
            input = input[:, ::-1]
            target = target[:, ::-1]
        if random.random()<0.5:
            input = input.transpose(1, 0)
            target = target.transpose(1, 0)
        return input, target


def random_crop(img, mask, patch_size, pos_prob=None):
    h, w = img.shape
    if min(h, w) < patch_size:
        img = np.pad(img, ((0, max(h, patch_size) - h), (0, max(w, patch_size) - w)), mode='constant')
        mask = np.pad(mask, ((0, max(h, patch_size) - h), (0, max(w, patch_size) - w)), mode='constant')
        h, w = img.shape

    cur_prob = random.random()
    if pos_prob == None or cur_prob > pos_prob or mask.max() == 0:
        h_start = random.randint(0, h - patch_size)
        w_start = random.randint(0, w - patch_size)
    else:
        loc = np.where(mask > 0)
        if len(loc[0]) <= 1:
            idx = 0
        else:
            idx = random.randint(0, len(loc[0]) - 1)
        h_start = random.randint(max(0, loc[0][idx] - patch_size), min(loc[0][idx], h - patch_size))
        w_start = random.randint(max(0, loc[1][idx] - patch_size), min(loc[1][idx], w - patch_size))

    h_end = h_start + patch_size
    w_end = w_start + patch_size
    img_patch = img[h_start:h_end, w_start:w_end]
    mask_patch = mask[h_start:h_end, w_start:w_end]

    return img_patch, mask_patch


def Normalized(img, img_norm_cfg):
    return (img - img_norm_cfg['mean']) / img_norm_cfg['std']


def Denormalization(img, img_norm_cfg):
    return img * img_norm_cfg['std'] + img_norm_cfg['mean']



# if __name__ == '__main__':
#     net = NestedUNet()
#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多
#     DATA = torch.randn(2, 3, 480, 480).to(DEVICE)
#     net.cuda()
#     output = net(DATA)
#     print("output:", np.shape(output))
#     total = sum([param.nelement() for param in net.parameters()])
#     print("Number of parameters: %.2fM" % (total / 1e6))


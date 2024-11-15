# -*- coding: utf-8 -*-
"""
@Time ： 2024/7/13 16:02
@Auth ： 归去来兮
@File ：train.py
@IDE ：PyCharm
@Motto:花中自幼微风起
"""

#from model.datasets_utils import NUDT_SIRST_train,NUDT_SIRST_test
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from torchvision import transforms
from model.utils import *
from model.Unet import Unet
import torch.optim as optim
from torch.optim import lr_scheduler
from model.Unet import Unet
from parse_args_mytrain import parse_args
if __name__ == "__main__":
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    mIoU=mIoU(1)
    ROC=ROCMetric(1,10)
    train_img_ids,test_img_ids,test_txt=load_dataset()
    mean_value = [0.2518, 0.2518, 0.2519]
    std_value = [0.2557, 0.2557, 0.2558]
    input_transform=transforms.Compose([
        #transforms.Resize((480,480)),
        transforms.ToTensor(),
        transforms.Normalize(mean_value,std_value)

    ]
    )

    train_set=TrainSetLoader(args.train_dataset_dir,train_img_ids,transform=input_transform)
    test_set=TestSetLoader(args.test_dataset_dir,test_img_ids,transform=input_transform)
    train_data =DataLoader(dataset=train_set, batch_size=1, shuffle=True, num_workers=4,drop_last=True)
    test_data = DataLoader(dataset=test_set, batch_size=1, shuffle=True, num_workers=4,drop_last=True)

    model=Unet()
    model=model.cuda()
    model.apply(weights_init_xavier)

    optimizer=optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=0.02)
    scheduler= lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)

    best_iou=0
    for epoch in range(args.epochs):
        tbar_train=tqdm(train_data)
        lr=scheduler.get_lr()[0]
        save_lr_dir = 'result_WS/' + '_learning_rate.log'
        with open(save_lr_dir, 'a') as f:
            f.write(' learning_rate: {:04f}:\n'.format(lr))
        print('learning_rate:', lr)
        model.train()
        losses=AverageMeter()
        for i,(data,labels) in enumerate(tbar_train):
            data = data.cuda()
            labels = labels.cuda()
            pred = model(data)
            loss = SoftIoULoss(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.update(loss.item(), pred.size(0))
            tbar_train.set_description('Epoch %d, training loss %.4f' % (epoch, losses.avg))
        train_loss=losses.avg
        scheduler.step()

        with torch.no_grad():
            tbar_test = tqdm(test_data)
            model.eval()
            mIoU.reset()
            for i, (data, labels) in enumerate(tbar_test):
                data = data.cuda()
                labels = labels.cuda()
                pred = model(data)
                loss = SoftIoULoss(pred, labels)
                losses.update(loss.item(), pred.size(0))
                ROC.update(pred, labels)
                mIoU.update(pred, labels)
                _, mean_IOU = mIoU.get()
                ture_positive_rate, false_positive_rate, recall, precision = ROC.get()
                tbar_test.set_description('Epoch %d, test loss %.4f, mean_IoU: %.4f' % (epoch, losses.avg, mean_IOU))
            test_loss=losses.avg

            save_train_test_loss_dir='result_WS'+'/train_test_loss.log'
            with open(save_train_test_loss_dir, 'a') as f:
                f.write('epoch: {:04f}:\t'.format(epoch))
                f.write('train_loss: {:04f}:\t'.format(train_loss))
                f.write('test_loss: {:04f}:\t'.format(test_loss))
                f.write('\n')

            if mean_IOU>best_iou:
                best_iou=mean_IOU
                save_model(best_iou, args.save_dir, args.save_prefix,
                           train_loss, test_loss, recall, precision, epoch, model.state_dict())














# -*- coding: utf-8 -*-
"""
@Time ： 2024/7/13 16:02
@Auth ： 归去来兮
@File ：detect.py
@IDE ：PyCharm
@Motto:花中自幼微风起
"""
# -*- coding: utf-8 -*-
"""
@Time ： 2024/7/13 16:02
@Auth ： 归去来兮
@File ：train.py
@IDE ：PyCharm
@Motto:花中自幼微风起
"""
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from torchvision import transforms
from model.Basic_utils import *
import torch.optim as optim
from torch.optim import lr_scheduler
from parse_args_mytest import parse_args
from model.Unet import Unet
from model.different_datasets import *
if __name__ == "__main__":
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    mIoU=mIoU(1)
    ROC=ROCMetric(1,10)
    train_img_ids,test_img_ids,test_txt=load_dataset()
    mean_value = [0.2518, 0.2518, 0.2519]
    std_value = [0.2557, 0.2557, 0.2558]
    input_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean_value,std_value)
    ]
    )
    # test_set=InferenceSetLoader(args.test_dataset_dir,test_img_ids,transform=input_transform)
    #
    # test_data = DataLoader(dataset=test_set, batch_size=1, shuffle=True, num_workers=4,drop_last=True)
    test_set = TestSetLoader(dataset_dir="D:\AweIRmodule", train_dataset_name="NUDT-SIRST",
                             test_dataset_name="NUDT-SIRST",
                             img_norm_cfg=dict(mean=107.80905151367188, std=33.02274703979492))
    test_data = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)

    best_iou=0
    epochs=10

    checkpoint = torch.load(args.result_WS+args.save_model+args.model_weight)
    model = Unet()
    model = model.cuda()
    model.load_state_dict(checkpoint['state_dict'])
    eval_image_path='result_WS/'+'vis_result'
    model.eval()
    tbar = tqdm(test_data)
    with torch.no_grad():
        num = 0
        for i, (data, size) in enumerate(tbar):
            data = data.cuda()
            pred = model(data)
            save_resize_pred(pred, size,crop_size=512,target_image_path=eval_image_path, val_img_ids=test_img_ids, num=num,suffix='.png')
            num+=1




















